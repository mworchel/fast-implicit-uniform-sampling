from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm

from utils import benchmark_single_call

from fast_implicit_uniform_sampling.sampling import sample_uniform_rays

def benchmark(method: str, num_rays: int, num_runs: int, num_warmup_runs: int, dim: int, output: Optional[Path] = None):
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a machine with a CUDA-capable GPU.")
        return
    
    ray_sampling_fn = get_ray_sampling_function(method)

    df = pd.DataFrame({'method': [], 'num_rays': [], 'run_idx': [], 'elapsed_time': [], 'peak_memory': [], 'success': []})

    # Warm up
    for _ in range(num_warmup_runs):
        with torch.no_grad():
            _, _, _ = ray_sampling_fn(num_rays, dim=dim, dtype=torch.float32, device=device)

    # Timed runs
    progress_range = tqdm(range(num_runs), leave=False)
    for run_idx in progress_range:
        progress_range.set_description_str(f"Run {run_idx+1}")
        torch.manual_seed(run_idx)

        elapsed_time_ms, peak_memory_mb, success = benchmark_single_call(ray_sampling_fn, num_rays, dim=dim, dtype=torch.float32, device=device)

        progress_range.set_postfix({"Elapsed": f"{elapsed_time_ms:.2f} ms", "Peak memory": f"{peak_memory_mb} MiB"})
        df.loc[len(df)] = [method, num_rays, run_idx, elapsed_time_ms, peak_memory_mb, success]

    if output is not None:
        df.to_csv(output, index=False)

    return df

def get_ray_sampling_function(method: str):
    if method == "iarap":
        from fast_implicit_uniform_sampling.sampling import sample_uniform_rays
        return sample_uniform_rays
    elif method == "ling":
        from ImplicitUniformSampler import ImplicitUniformSampler
        instance = ImplicitUniformSampler()
        def sample_rays(num_rays: int, dim: int = 3, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
            if dim != 3:
                raise ValueError("Only 3D sampling is supported for the 'ling' method.")
            return instance.sample_rays(num_rays)
        return sample_rays
    elif method == "ling_pytorch":
        from ling_pytorch import sample_rays
        return sample_rays
    else:
        raise ValueError(f"Unknown method: {method}")
    
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, required=True, choices=["iarap", "ling", "ling_pytorch"], help="Method to use for uniform sampling")
    parser.add_argument("--num_rays", type=int, default=10000, help="Number of rays to sample")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--num_warmup_runs", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--dim", type=int, default=3, help="Dimension of the rays to sample (only 3D is supported for 'ling' method)")
    parser.add_argument("--output", type=Path, default=None, help="Output file for the benchmark results")
    args = parser.parse_args()

    print(f"Benchmarking method '{args.method}' with {args.num_rays} rays.")

    df = benchmark(method=args.method, num_rays=args.num_rays, num_runs=args.num_runs, num_warmup_runs=args.num_warmup_runs, dim=args.dim, output=args.output)
    if args.output is None:
        print(df)