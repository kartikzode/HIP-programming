import argparse
import importlib.util
from pathlib import Path

import torch


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark(fn, data, warmup: int = 10, iterations: int = 50):
    for _ in range(warmup):
        fn(data)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn(data)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description="Compare latency of the two MoE kernel approaches")
    parser.add_argument("--d_hidden", type=int, default=512)
    parser.add_argument("--d_expert", type=int, default=128)
    parser.add_argument("--n_routed_experts", type=int, default=8)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--n_experts_per_token", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=81934)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.cuda.set_device(0)
    base_dir = Path(__file__).resolve().parent

    module_00 = load_module("moe_00", base_dir / "00.py")
    module_02 = load_module("moe_02", base_dir / "02.py")

    input_tensor, weights, config = module_00.generate_input(
        args.d_hidden,
        args.d_expert,
        args.n_routed_experts,
        args.n_shared_experts,
        args.n_experts_per_token,
        args.batch_size,
        args.seq_len,
        args.seed,
    )

    data = (input_tensor, weights, config)
    out_00 = module_00.custom_kernel(data)
    out_02 = module_02.custom_kernel(data)

    match = torch.allclose(out_00, out_02, atol=1e-2, rtol=0)
    print(f"Output match: {match}")

    latency_00 = benchmark(module_00.custom_kernel, data, warmup=args.warmup, iterations=args.iterations)
    latency_02 = benchmark(module_02.custom_kernel, data, warmup=args.warmup, iterations=args.iterations)

    print(f"Approach 00 latency: {latency_00:.4f} ms")
    print(f"Approach 02 latency: {latency_02:.4f} ms")
    print(f"Speedup (02 over 00): {latency_00 / latency_02:.2f}x")


if __name__ == "__main__":
    main()
