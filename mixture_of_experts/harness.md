# MoE kernel test/bench harness

A trimmed-down version of the gpumode `eval.py` harness, tailored for developing
locally on a rented GPU rather than running inside the competition's queue
infrastructure. Same correctness checks, same shapes, same geometric-mean
scoring rule (`ranking_by: geom` in the original yaml) — minus the
`POPCORN_FD` / multiprocessing-pool / leaderboard-submission plumbing that
only matters for the actual competition runner.

## Files

| file | purpose |
|---|---|
| `task.py` | shared type definitions (`input_t`, `output_t`, `TestSpec`) |
| `reference.py` | `generate_input()`, a simple/obviously-correct PyTorch `ref_kernel()`, and `check_implementation()` |
| `utils.py` | unchanged from the original repo (`verbose_allclose`, `set_seed`, etc.) |
| `bench.py` | the CLI: correctness tests, benchmarks vs. the PyTorch baseline, geomean report |
| `submission_example.py` | a working (unoptimized) starting point — just wraps `ref_kernel` — so you can confirm the harness runs end-to-end before writing any Triton |

## Setup on the AMD box

Put all five files in one directory, plus your own kernel as `submission.py`
(it just needs a top-level `custom_kernel(data: input_t) -> output_t`
function — see `submission_example.py` for the exact contract).

```bash
# sanity-check the harness itself first:
python bench.py --submission submission_example.py

# then, once you've got real Triton code:
python bench.py                          # uses ./submission.py by default
```

## What it does

1. **Correctness** — runs your `custom_kernel` against the 3 shapes from the
   yaml's `tests:` list, checks each output against `ref_kernel` with
   `rtol=atol=1e-2` (same tolerance used in the ad-hoc checks found inside
   the reference submissions themselves — tight enough to catch real bugs,
   loose enough to not flag ordinary fp16 rounding-order differences).
2. **Benchmarks** — runs your kernel against the 2 shapes from the yaml's
   `benchmarks:` list (`n_routed_experts=32`, `seq_len=2048` and `8192`),
   timing with the same warmup + repeat-until-stable rule as the original
   (`torch.cuda.synchronize()` around each call, stop once the relative
   standard error drops below target or the repeat/time budget runs out).
   The PyTorch reference gets timed the same way alongside it, so you get a
   direct speedup number per shape.
3. **Geomean** — the final line is the geometric mean of your kernel's mean
   latency across the two benchmark shapes — this is the actual number the
   real leaderboard optimizes (`ranking_by: "geom"`), so it's what you should
   be watching go down as you iterate.

## Useful flags

- `--test-only` / `--bench-only` — run just one half.
- `--quick` — fewer repeats and a looser stopping threshold, for a fast
  edit-run-edit loop while you're still actively changing the kernel. Switch
  back to the default (no flag) before trusting a number.
- `--no-baseline` — skip timing `ref_kernel`, if you just want your own
  kernel's numbers faster.
- `--submission path/to/file.py` — point at any file, doesn't have to be
  literally named `submission.py`.

## A couple of things worth knowing

- **`device="cuda"` is correct for ROCm too.** PyTorch's ROCm build keeps the
  `cuda` device string for compatibility, so nothing here needs changing to
  run on the MI300X.
- **Data gets cloned before every correctness check and before every timed
  call** (`_clone_data`), because some of the faster kernel designs
  pre-transpose or restack the weights dict into scratch buffers on their
  first touch — cloning means neither the correctness baseline nor
  subsequent benchmark repeats can be silently corrupted by that.
- If you want the harness to also test with different `n_routed_experts` /
  `seq_len` combinations beyond what's in the yaml, just add entries to
  `TEST_SHAPES` / `BENCHMARK_SHAPES` at the top of `bench.py` — every entry
  is a plain dict matching `generate_input`'s kwargs.