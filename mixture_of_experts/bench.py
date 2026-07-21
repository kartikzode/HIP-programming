"""
Local test + benchmark harness for the AMD MoE kernel task, tailored from the
gpumode eval.py for running by hand on a rented GPU (no POPCORN_FD, no
multiprocessing sandbox, no leaderboard-submission plumbing -- just: does it
pass, and how fast is it vs. the PyTorch baseline).

Usage
-----
    python bench.py                          # correctness tests + benchmarks
    python bench.py --test-only              # just the 3 `tests:` shapes
    python bench.py --bench-only             # just the 2 `benchmarks:` shapes
    python bench.py --submission my_kernel.py
    python bench.py --quick                  # fewer repeats, for fast iteration
    python bench.py --no-baseline            # skip timing the PyTorch reference

Shapes are copied directly from the competition yaml's `tests:` and
`benchmarks:` lists. Final benchmark score is reported as a geometric mean
across the `benchmarks:` shapes, matching `ranking_by: "geom"` in the yaml.
"""
import argparse
import dataclasses
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch

from reference import check_implementation, generate_input, ref_kernel
from task import TestSpec
from utils import set_seed

# ---------------------------------------------------------------------------
# Shapes, copied straight from the competition yaml.
# ---------------------------------------------------------------------------

TEST_SHAPES: List[TestSpec] = [
    dict(dhidden=7168, dexpert=2048, nroutedexperts=4, nsharedexperts=1,
         nexpertspertoken=4, bs=1, seqlen=512, seed=9371),
    dict(dhidden=7168, dexpert=2048, nroutedexperts=8, nsharedexperts=1,
         nexpertspertoken=4, bs=2, seqlen=512, seed=2291),
    dict(dhidden=7168, dexpert=2048, nroutedexperts=8, nsharedexperts=1,
         nexpertspertoken=4, bs=1, seqlen=8192, seed=81934),
] # type: ignore

BENCHMARK_SHAPES: List[TestSpec] = [
    dict(dhidden=7168, dexpert=2048, nroutedexperts=32, nsharedexperts=1,
         nexpertspertoken=4, bs=1, seqlen=2048, seed=9371),
    dict(dhidden=7168, dexpert=2048, nroutedexperts=32, nsharedexperts=1,
         nexpertspertoken=4, bs=1, seqlen=8192, seed=1212),
]  # type: ignore


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clone_data(data):
    """Recursively clone all tensors in (input, weights_dict, config). Some
    kernels pre-transpose/stack weights into scratch buffers and could in
    principle mutate what they're handed, so every timed/checked call gets
    its own copy -- same reasoning as the original eval.py."""
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def load_submission(path: str) -> Callable:
    """Dynamically import `custom_kernel` from an arbitrary file path, so you
    don't have to literally name your kernel file submission.py."""
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_kernel"):
        raise AttributeError(f"{path} has no top-level `custom_kernel(data)` function")
    return module.custom_kernel


def shape_label(spec: TestSpec) -> str:
    return (f"E={spec['nroutedexperts']:<3d} topk={spec['nexpertspertoken']} "
            f"bs={spec['bs']} seq={spec['seqlen']:<5d} d_hidden={spec['dhidden']} "
            f"d_expert={spec['dexpert']}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    runs: int
    mean_ns: float
    std_ns: float
    err_ns: float
    best_ns: float
    worst_ns: float

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / 1e6


def calculate_stats(durations_ns: List[int]) -> Stats:
    runs = len(durations_ns)
    avg = sum(durations_ns) / runs
    variance = sum((d - avg) ** 2 for d in durations_ns) / max(runs - 1, 1)
    std = math.sqrt(variance)
    err = std / math.sqrt(runs)
    return Stats(runs=runs, mean_ns=avg, std_ns=std, err_ns=err,
                 best_ns=float(min(durations_ns)), worst_ns=float(max(durations_ns)))


def time_kernel(
    kernel_fn: Callable,
    shape: TestSpec,
    min_repeats: int = 5,
    max_repeats: int = 100,
    max_total_time_ns: float = 10e9,
    rel_err_target: float = 0.001,
    wallclock_budget_s: float = 120.0,
) -> Stats:
    """Same stopping rules as the original _run_single_benchmark: at least
    min_repeats runs, stop early once the relative standard error of the
    mean drops below rel_err_target, otherwise stop once max_repeats or the
    time/wallclock budget is hit. Data is regenerated once and reused for
    every repeat (matches the non-`recheck` leaderboard path)."""
    data = generate_input(**shape)

    # warmup: also pays for autotuning / triton's first-launch compilation
    _ = kernel_fn(_clone_data(data))
    torch.cuda.synchronize()

    durations = []
    wall_start = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        _ = kernel_fn(data)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        durations.append(t1 - t0)

        if i + 1 >= min_repeats:
            stats = calculate_stats(durations)
            if stats.err_ns / stats.mean_ns < rel_err_target:
                break
            if stats.mean_ns * stats.runs > max_total_time_ns:
                break
        if (time.perf_counter_ns() - wall_start) > wallclock_budget_s * 1e9:
            break

    return calculate_stats(durations)


def geomean(values: List[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def run_correctness(custom_kernel: Callable, shapes: List[TestSpec]) -> bool:
    print("=" * 100)
    print("CORRECTNESS TESTS")
    print("=" * 100)
    all_good = True
    for i, shape in enumerate(shapes):
        data = generate_input(**shape)
        check_copy = _clone_data(data)
        try:
            output = custom_kernel(_clone_data(data))
        except Exception as e:  # noqa: BLE001 - want to report and keep going
            print(f"[test {i}] {shape_label(shape)}")
            print(f"    CRASHED: {type(e).__name__}: {e}")
            all_good = False
            continue

        good, msg = check_implementation(check_copy, output)
        print(f"[test {i}] {shape_label(shape)} -> {'PASS' if good else 'FAIL'}")
        if not good:
            for line in msg.splitlines():
                print(f"    {line}")
            all_good = False
        elif msg:
            print(f"    {msg.splitlines()[0]}")
    print()
    return all_good


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def run_benchmarks(
    custom_kernel: Callable,
    shapes: List[TestSpec],
    quick: bool,
    with_baseline: bool,
) -> None:
    print("=" * 100)
    print("BENCHMARKS" + ("  (--quick: fewer repeats, noisier numbers)" if quick else ""))
    print("=" * 100)

    kwargs = dict(min_repeats=3, max_repeats=10, rel_err_target=0.02) if quick else {}

    header = f"{'shape':70s} {'submission (ms)':>16s}"
    if with_baseline:
        header += f" {'pytorch ref (ms)':>17s} {'speedup':>9s}"
    print(header)
    print("-" * len(header))

    sub_means, ref_means = [], []
    for shape in shapes:
        sub_stats = time_kernel(custom_kernel, shape, **kwargs)
        sub_means.append(sub_stats.mean_ns)
        row = f"{shape_label(shape):70s} {sub_stats.mean_ms:16.4f}"

        if with_baseline:
            ref_stats = time_kernel(ref_kernel, shape, **kwargs)
            ref_means.append(ref_stats.mean_ns)
            speedup = ref_stats.mean_ns / sub_stats.mean_ns
            row += f" {ref_stats.mean_ms:17.4f} {speedup:8.2f}x"
        print(row)

    print("-" * len(header))
    sub_gm = geomean(sub_means)
    gm_row = f"{'GEOMEAN (this is the leaderboard score, ranking_by=geom)':70s} {sub_gm/1e6:16.4f}"
    if with_baseline:
        ref_gm = geomean(ref_means)
        gm_row += f" {ref_gm/1e6:17.4f} {ref_gm/sub_gm:8.2f}x"
    print(gm_row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", default="submission.py",
                         help="path to your kernel file (must define custom_kernel(data)). Default: submission.py")
    parser.add_argument("--test-only", action="store_true", help="only run correctness tests")
    parser.add_argument("--bench-only", action="store_true", help="only run benchmarks (skips correctness gate)")
    parser.add_argument("--quick", action="store_true", help="fewer repeats / looser stopping rule, for fast iteration")
    parser.add_argument("--no-baseline", action="store_true", help="don't time the PyTorch reference alongside your kernel")
    parser.add_argument("--seed", type=int, default=42, help="global seed (default matches original harness: 42)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA (or ROCm-as-CUDA) device visible. On the AMD box, make sure you're in the right "
              "container/env -- `torch.cuda.is_available()` should be True under a ROCm PyTorch build too.",
              file=sys.stderr)
        return 111

    print(f"device: {torch.cuda.get_device_name(0)}")
    set_seed(args.seed)

    custom_kernel = load_submission(args.submission)

    if not args.bench_only:
        ok = run_correctness(custom_kernel, TEST_SHAPES)
        if not ok and not args.test_only:
            print("Correctness tests failed -- benchmarking anyway, but take the numbers with a grain of salt.\n")

    if args.test_only:
        return 0

    run_benchmarks(custom_kernel, BENCHMARK_SHAPES, quick=args.quick, with_baseline=not args.no_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())