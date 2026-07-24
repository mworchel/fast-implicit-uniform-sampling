from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from pathlib import Path
from skimage.measure import marching_cubes
from typing import Optional
import torch
from tqdm import tqdm

from dataset import Dataset
from geometry import project_points_to_mesh, compute_face_areas

from fast_implicit_uniform_sampling.sampling import sample_uniform_rays

from benchmark_uniform_point_sampling import get_sampler_class

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, required=True, choices=["iarap", "ling", "reference"], help="Method to use for uniform sampling")
    parser.add_argument("--dataset", type=str, required=True, choices=["neural_iarap"], help="Dataset to use for benchmarking")
    parser.add_argument("--sdf_filter", type=str, default=None, nargs="+", help="Optional filter to select specific SDFs from the dataset")
    parser.add_argument("--num_rays", type=int, default=10000, help="Number of rays to sample")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--output", type=Path, default=None, help="Output file for the results")
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    print(f"Measuring sample uniformity for method '{args.method}' on dataset '{args.dataset}' with {args.num_rays} rays.")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a machine with a CUDA-capable GPU.")
        return
    
    dataset = Dataset(args.dataset) 
    print(f"Dataset '{args.dataset}' contains {len(dataset)} SDFs.")

    if args.sdf_filter is not None:
        print(f"Filter: only running benchmark with SDFs {args.sdf_filter}")

    sampler = get_sampler_class(args.method)(device=device, dtype=torch.float32) if args.method != "reference" else UniformMeshSampler()

    df = pd.DataFrame({'method': [], 'dataset': [], 'sdf_name': [], 'run_idx': [], 'num_samples': [], 'total_variation': [], 'success': []})
    for idx in range(len(dataset)):
        sdf_name = dataset.sdf_paths[idx].stem
        print(f"-- {idx+1}/{len(dataset)}: {sdf_name}")
        sdf = dataset[idx].to(device).eval()

        if args.sdf_filter is not None and sdf_name not in args.sdf_filter:
            continue

        # Extract "ground truth" mesh from the SDF (or use cached version if available)
        res = 128
        cached_mesh_path = cache_dir / f"{args.dataset}_{sdf_name}_{res}_mesh.npz"
        if cached_mesh_path.exists():
            print(f"Loading cached mesh from {cached_mesh_path}")
            mesh_data = torch.load(cached_mesh_path, map_location=device)
            v, f = mesh_data['vertices'], mesh_data['faces']
        else:
            print(f"Extracting mesh from SDF at resolution {res}...")
            v, f = extract_mesh_from_sdf(sdf, res, device)
            torch.save({'vertices': v, 'faces': f}, cached_mesh_path)
        
        # # Debug: write out mesh to obj
        # with open(f"{sdf_name}_mesh.obj", "w") as file:
        #     for vertex in v:
        #         file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        #     for face in f + 1:  # OBJ format uses 1-based indexing
        #         file.write(f"f {face[0]} {face[1]} {face[2]}\n")

        if args.method == "reference":
            sampler.set_mesh(v, f)

        tv_metric = TotalVariationMetric(v, f)

        progress_range = tqdm(range(args.num_runs), leave=False)
        for run_idx in progress_range:
            progress_range.set_description_str(f"Run {run_idx+1}")

            torch.manual_seed(idx ^ run_idx ^ args.num_rays)

            try:
                with torch.no_grad():
                    samples = sampler.sample(sdf, num_rays=args.num_rays)
            except Exception as e:
                print(f"Error during sampling for SDF '{sdf_name}' on run {run_idx}: {e}")
                df.loc[len(df)] = [args.method, args.dataset, sdf_name, run_idx, None, None, False]
                continue
            
            # # Debug: write out sample points to obj
            # with open(f"{sdf_name}_samples.obj", "w") as file:
            #     for point in samples:
            #         file.write(f"v {point[0]} {point[1]} {point[2]}\n")

            df.loc[len(df)] = [args.method, args.dataset, sdf_name, run_idx, len(samples), tv_metric(samples), True]

    if args.output is not None:
        df.to_csv(args.output, index=False)
    else:
        print(df)

def extract_mesh_from_sdf(sdf, res: int = 128, device: Optional[torch.device] = None):
    """
    Extract a triangle mesh from an SDF using the marching cubes algorithm.
    """
    grid = torch.linspace(-1, 1, res, device=device)
    X, Y, Z = torch.meshgrid(grid, grid, grid, indexing='ij')
    points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)
    spacing = (2.0 / (res - 1), 2.0 / (res - 1), 2.0 / (res - 1))
    with torch.no_grad():
        sdf_values = sdf(points).reshape(res, res, res).cpu().numpy()
        v, f, _, _ = marching_cubes(sdf_values, level=0.0, spacing=spacing)
        v -= 1.0  # Adjust vertices to be in the range [-1, 1]
    v = torch.from_numpy(v).float().to(device)
    f = torch.from_numpy(f.copy()).long().to(device)
    return v, f

class UniformMeshSampler:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.face_areas = None

    def set_mesh(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces
        self.face_areas = compute_face_areas(vertices, faces)
        self.face_areas /= self.face_areas.sum()

    def sample(self, sdf: torch.nn.Module, num_rays: int) -> torch.Tensor:
        """
        Sample points uniformly on the mesh surface.
        """
        # SDF is unused (use cached mesh instead)

        face_indices = torch.multinomial(self.face_areas, num_samples=num_rays, replacement=True)
        v0, v1, v2 = self.vertices[self.faces[face_indices]].unbind(1)

        # Sample uniformly within each triangle
        u = torch.rand(num_rays, device=self.vertices.device)[:, None]
        v = torch.rand(num_rays, device=self.vertices.device)[:, None]

        mask = (u + v > 1)
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]

        samples = (v0 * (1 - u - v) + v1 * u + v2 * v)
        return samples

class TotalVariationMetric:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces
        self.face_areas = compute_face_areas(vertices, faces)
        self.face_areas /= self.face_areas.sum()

    def __call__(self, points: torch.Tensor) -> float:
        try:
            import igl
            _, samples_face_idx, _ = igl.point_mesh_squared_distance(points.cpu().numpy(), self.vertices.cpu().numpy(), self.faces.cpu().numpy())
            samples_face_idx = torch.from_numpy(samples_face_idx).to(points.device)
        except ImportError:
            # Naive fallback if igl is not available
            _, _, samples_face_idx = project_points_to_mesh(points, self.vertices, self.faces, chunk_size=1000)
        #face_idx, face_num_samples = torch.unique(samples_face_idx, return_counts=True)

        face_num_samples = torch.zeros_like(self.face_areas)
        face_num_samples.scatter_add_(0, samples_face_idx, torch.ones_like(samples_face_idx, dtype=torch.float32))

        return 0.5 * (torch.abs(self.face_areas - face_num_samples / len(points))).sum().item()

if __name__ == "__main__":
    main()