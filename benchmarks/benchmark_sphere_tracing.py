from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm

from dataset import Dataset
from utils import benchmark_single_call

from fast_implicit_uniform_sampling.sampling import sample_uniform_rays

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, required=True, choices=["iarap", "ling"], help="Method to use for uniform sampling")
    parser.add_argument("--dataset", type=str, required=True, choices=["neural_iarap"], help="Dataset to use for benchmarking")
    parser.add_argument("--sdf_filter", type=str, default=None, nargs="+", help="Optional filter to select specific SDFs from the dataset")
    parser.add_argument("--num_rays", type=int, default=10000, help="Number of rays to sample")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--ray_split_threshold", type=float, default=None, help="Length threshold for ray splitting (optional, only for `iarap` method)")
    parser.add_argument("--output", type=Path, default=None, help="Output file for the benchmark results")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag stored with the benchmark results")
    args = parser.parse_args()

    print(f"Benchmarking method '{args.method}' on dataset '{args.dataset}' with {args.num_rays} rays.")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a machine with a CUDA-capable GPU.")
        return
    
    dataset = Dataset(args.dataset) 
    print(f"Dataset '{args.dataset}' contains {len(dataset)} SDFs.")

    if args.sdf_filter is not None:
        print(f"Filter: only running benchmark with SDFs {args.sdf_filter}")

    sphere_tracing_fn = get_sphere_tracing_function(args.method)

    # Assemble kwargs for the sphere tracing function
    sphere_tracing_kwargs = {}
    if args.method == "iarap" and args.ray_split_threshold is not None:
        sphere_tracing_kwargs["ray_split_threshold"] = args.ray_split_threshold

    df = pd.DataFrame({'method': [], 'num_rays': [], 'dataset': [], 'sdf_name': [], 'run_idx': [], 'elapsed_time': [], 'peak_memory': [], 'tag': [], 'success': []})
    for idx in range(len(dataset)):
        sdf_name = dataset.sdf_paths[idx].stem
        print(f"-- {idx+1}/{len(dataset)}: {sdf_name}")
        sdf = dataset[idx].to(device).eval()

        if args.sdf_filter is not None and sdf_name not in args.sdf_filter:
            continue

        # Warm-up
        torch.manual_seed(idx ^ 0)
        ray_o, ray_d, max_t = sample_uniform_rays(args.num_rays, dim=3, dtype=torch.float32, device=device)
        _, _, success = benchmark_single_call(sphere_tracing_fn, sdf, ray_o, ray_d, max_t, **sphere_tracing_kwargs)

        # If warm-up failed, skip the timed runs and store invalid data
        if not success:
            for run_idx in range(args.num_runs):
                df.loc[len(df)] = [args.method, args.num_rays, args.dataset, sdf_name, run_idx, None, None, args.tag, False]
            continue

        # Timed runs
        progress_range = tqdm(range(args.num_runs), leave=False)
        for run_idx in progress_range:
            progress_range.set_description_str(f"Run {run_idx+1}")

            torch.manual_seed(idx ^ run_idx)
            ray_o, ray_d, max_t = sample_uniform_rays(args.num_rays, dim=3, dtype=torch.float32, device=device)

            elapsed_time_ms, peak_memory_mb, success = benchmark_single_call(sphere_tracing_fn, sdf, ray_o, ray_d, max_t, **sphere_tracing_kwargs)

            progress_range.set_postfix({"Elapsed": f"{elapsed_time_ms:.2f} ms", "Peak memory": f"{peak_memory_mb} MiB"})

            df.loc[len(df)] = [args.method, args.num_rays, args.dataset, sdf_name, run_idx, elapsed_time_ms, peak_memory_mb, args.tag, success]

    if args.output is not None:
        df.to_csv(args.output, index=False)
    else:
        print(df)

def get_sphere_tracing_function(method: str):
    if method == "iarap":
        from fast_implicit_uniform_sampling.sampling import sphere_trace_all_intersections
        return sphere_trace_all_intersections
    elif method == "ling":
        from ImplicitUniformSampler.sampler import sphere_trace_modified
        def sphere_trace_modified_wrapper(sdf, ray_origin, ray_direction, max_t):
            return sphere_trace_modified(ray_origin, ray_direction, lambda x: sdf(x).squeeze(-1), max_t)[0]
        return sphere_trace_modified_wrapper
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    main()