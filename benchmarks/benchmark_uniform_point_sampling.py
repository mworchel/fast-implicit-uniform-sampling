from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import pandas as pd
from typing import Optional
import torch
from tqdm import tqdm

from dataset import Dataset
from utils import benchmark_single_call

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, required=True, choices=["iarap", "ling", "ling_pytorch"], help="Method to use for uniform sampling")
    parser.add_argument("--dataset", type=str, required=True, choices=["neural_iarap"], help="Dataset to use for benchmarking")
    parser.add_argument("--sdf_filter", type=str, default=None, nargs="+", help="Optional filter to select specific SDFs from the dataset")
    parser.add_argument("--num_rays", type=int, default=10000, help="Number of rays to sample")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--output", type=Path, default=None, help="Output file for the benchmark results")
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

    sampler = get_sampler_class(args.method)(device=device, dtype=torch.float32)

    df = pd.DataFrame({'method': [], 'num_rays': [],  'dataset': [], 'sdf_name': [], 'run_idx': [], 'elapsed_time': [], 'peak_memory': [], 'success': []})
    for idx in range(len(dataset)):
        sdf_name = dataset.sdf_paths[idx].stem
        print(f"-- {idx+1}/{len(dataset)}: {sdf_name}")
        sdf = dataset[idx].to(device).eval()

        if args.sdf_filter is not None and sdf_name not in args.sdf_filter:
            continue

        # Warm up
        with torch.no_grad():
            _, _, success = benchmark_single_call(sampler.sample, sdf, num_rays=args.num_rays)

        # If warm-up failed, skip the timed runs and store invalid data
        if not success:
            for run_idx in range(args.num_runs):
                df.loc[len(df)] = [args.method, args.num_rays, args.dataset, sdf_name, run_idx, None, None, False]
            continue

        # Timed runs
        progress_range = tqdm(range(args.num_runs), leave=False)
        for run_idx in progress_range:
            progress_range.set_description_str(f"Run {run_idx+1}")

            elapsed_time_ms, peak_memory_mb, success = benchmark_single_call(sampler.sample, sdf, num_rays=args.num_rays)
            
            progress_range.set_postfix({"Elapsed": f"{elapsed_time_ms:.2f} ms", "Peak memory": f"{peak_memory_mb} MiB"})
            
            df.loc[len(df)] = [args.method, args.num_rays, args.dataset, sdf_name, run_idx, elapsed_time_ms, peak_memory_mb, success]

    if args.output is not None:
        df.to_csv(args.output, index=False)
    else:
        print(df)

def get_sampler_class(method: str):
    if method == "iarap":
        from fast_implicit_uniform_sampling import ImplicitUniformSampler
        return ImplicitUniformSampler
    elif method in ["ling", "ling_pytorch"]:
        from ImplicitUniformSampler.sampler import sphere_trace_modified
        from ImplicitUniformSampler import ImplicitUniformSampler
        import ling_pytorch

        # Wrap the original sampler to support different devices and data types
        class ImplicitUniformSamplerWrapper(ImplicitUniformSampler):
            def __init__(self, thresh: float = 1e-4, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
                    super().__init__(thresh=thresh)
                    self.device = device
                    self.dtype = dtype

            def sample(self, sdf_func, num_rays): 
                dtype  = self.dtype
                device = self.device
                # Try to infer the dtype and device using the module parameters (if not given)
                if self.dtype is None or self.device is None:
                    if isinstance(sdf_func, torch.nn.Module):
                        param = next(sdf_func.parameters())
                        dtype  = param.dtype
                        device = param.device
                    else:
                        raise RuntimeError("""Unable to infer dtype from the SDF function (it is not a torch.nn.Module).
                                            Please provide device and dtype to the sampler's constructor.""")

                sdf_fn_squeeze = lambda x: sdf_func(x).squeeze(-1)  # Ensure the output is of shape (N,) for N input points

                # This is the original sampler code (but it uses device and dtype from the wrapper)

                if method == "ling":
                    lines_origins, lines_dirs, max_ts = self.sample_rays(num_rays)     
                    lines_origins = torch.from_numpy(lines_origins).to(device=device, dtype=dtype)
                    lines_dirs = torch.from_numpy(lines_dirs).to(device=device, dtype=dtype)
                    max_ts = torch.from_numpy(max_ts).to(device=device, dtype=dtype)
                else: # method == ling_pytorch
                    # Use our PyTorch port to avoid the CPU and data transfers
                    lines_origins, lines_dirs, max_ts = ling_pytorch.sample_rays(num_rays, dim=3, dtype=dtype, device=device)

                pts, _ = sphere_trace_modified(lines_origins, 
                                               lines_dirs, 
                                               sdf_fn=sdf_fn_squeeze, 
                                               max_t=max_ts,
                                               eps=self.thresh) 
                return pts 
        return ImplicitUniformSamplerWrapper
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    main()