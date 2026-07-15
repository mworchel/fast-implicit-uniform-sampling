import torch
from typing import Callable, Optional

@torch.no_grad()
def sample_uniform_rays(num_rays: int, dim: int = 3, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    """
    Uniformely sample rays in the [-1,1]^dim cube.
    The ray origins lie on the faces of the cube and the direction points inwards.

    Parameters
    ----------
    num_rays : int
        Number of rays to sample.
    dim : int
        The dimensionality of the space (can be 2 or 3).
    dtype : Optional[torch.dtype]
        The data type of the returned rays (uses torch default if not given).
    device : Optional[torch.device]
        The device used for the sampling procedure (uses torch default if not given).
        
    Returns
    -------
    origins : torch.Tensor
        Origins of the sampled rays, as an array of shape (num_rays, dim).
    directions : torch.Tensor
        Normalized directions of the sampled rays, as an array of shape (num_rays, dim).
    t_max : torch.Tensor
        Distance along the rays from their origin to their exit face of the cube, as an array of shape (num_rays,).
    """
    if num_rays < 0:
        raise ValueError("Number of sampled rays must not be negative.")

    if dim not in [2, 3]:
        raise NotImplementedError("Only dim=2 and dim=3 are supported")

    # 1. Uniformely sample a direction on the unit sphere
    d = torch.randn(num_rays, dim, dtype=dtype, device=device)
    d = torch.nn.functional.normalize(d, dim=-1)  # (N, dim)

    # 2. Determine the sign of the normal for the visible faces.
    #    For each axis (x,y,z), only one face is visible from the direction `d` at any time (modulo degenerate cases).
    #    A face with normal `n` is visible from direction `d` if `d^T n > 0`.
    face_signs = torch.sign(d) # (N,dim) +1 or -1

    # 3. Sample one of the three visible faces with probability proportional to its projected area.
    #    The weight for a face i is simply |d^T n_i| = |d_i| since the normals are axis-aligned.
    weights = d.abs() # (N, dim)
    axis_idx = torch.multinomial(weights, num_samples=1).squeeze(-1)  # (N,)

    # 4. Uniformly sample a point on the chosen face
    #    The sample represents coordinates with respect to the other axes, the 'free' axes.
    #    Example: for a fixed axis 0, the free axes are (1,2).
    u = 2 * torch.rand(num_rays, dim-1, dtype=dtype, device=device) - 1
    if dim == 2:
        free_axes = torch.tensor([[1], [0]], device=device)
    else: # dim == 3:
        free_axes = torch.tensor([[1, 2], [0, 2], [0, 1]], device=device)

    # 5. Assemble the ray origins
    ray_idx = torch.arange(num_rays, device=device) # (N,)
    origin = torch.empty(num_rays, dim, dtype=dtype, device=device)
    origin[ray_idx, axis_idx] = face_signs[ray_idx, axis_idx]
    origin[ray_idx, free_axes[axis_idx, 0]] = u[:, 0]
    if dim == 3:
        origin[ray_idx, free_axes[axis_idx, 1]] = u[:, 1]

    # 6. Make sure the direction points inwards
    direction = -d
    
    # 7. Determine the maximum ray length (minimum distance to the exit faces).
    t_exit_faces = (-face_signs - origin) / direction # (N, dim)
    t_max = t_exit_faces.min(dim=-1).values # (N,)

    return origin, direction, t_max

@torch.no_grad()
def split_rays(origin: torch.Tensor, direction: torch.Tensor, t: torch.Tensor, max_t: torch.Tensor, mask: torch.Tensor):
    mask_nonzero = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    t_mid = 0.5 * (t[mask_nonzero] + max_t[mask_nonzero])

    dt = (t_mid - t[mask_nonzero]).unsqueeze(-1)
    origin_second = origin[mask_nonzero] + dt * direction[mask_nonzero]
    direction_second = direction[mask_nonzero]

    return mask_nonzero, origin_second, direction_second, t_mid

@torch.no_grad()
def sphere_trace_all_intersections(sdf: Callable[[torch.Tensor], torch.Tensor], 
                                   ray_origin: torch.Tensor, ray_direction: torch.Tensor, max_t: torch.Tensor, 
                                   eps: float = 1e-4, step_bound: float = 10,
                                   ray_split_threshold: float = 0.01, # Only split rays longer than this threshold
                                   ray_budgeting: str = "soft", # "soft" or "hard" ray budgeting
                                   verbose: bool = False):
    """
    Sphere trace rays against an SDF, returning all intersections with an epsilon band around the surface.
    """

    dtype = ray_origin.dtype
    device = ray_origin.device 

    ray_budget = 1e6 # Soft maximum number of rays in flight at any time
    max_iters  = 2000 # Maximum number of iterations per ray (same as Ling et al.)

    eps_half = torch.tensor(eps * 0.5, dtype=dtype, device=device)
    
    ray_origin = ray_origin.to(dtype=dtype, device=device)
    ray_direction = ray_direction.to(dtype=dtype, device=device)
    max_t = max_t.to(dtype=dtype, device=device)

    p        = ray_origin.clone()
    d        = ray_direction.clone()
    t        = torch.zeros_like(max_t)
    iters    = torch.zeros_like(max_t)
    crossing = torch.full_like(max_t, False, dtype=torch.bool)
    active_idx   = torch.arange(ray_origin.shape[0], device=device)
    hit_chunks = []
    while active_idx.numel() > 0:
        # Split non-crossing rays
        n_base = p.shape[0]
        n_split = ray_budget - n_base
        split = n_split > 0 if ray_budgeting == "hard" else n_base < (0.6 * ray_budget)
        if split:
            is_long_ray = (max_t - t) > ray_split_threshold if ray_split_threshold > 0 else True
            split_idx, p_split, d_split, t_split = split_rays(p, d, t, max_t, mask=~crossing & is_long_ray)

            if ray_budgeting == "hard":
                # Reject splits exceeding the ray budget (just take the first `n_split` rays)
                split_idx = split_idx[:n_split]
                p_split = p_split[:n_split]
                d_split = d_split[:n_split]
                t_split = t_split[:n_split]

            p = torch.cat([p, p_split], dim=0)
            d = torch.cat([d, d_split], dim=0)
            t = torch.cat([t, t_split], dim=0)
            crossing = torch.cat([crossing, torch.full_like(t_split, False, dtype=torch.bool)], dim=0)
            max_t = torch.cat([max_t, max_t[split_idx]], dim=0) # This is modified later when a split is accepted
            iters = torch.cat([iters, iters[split_idx]], dim=0)
        elif verbose:
            print(f"Ray budget exceeded: {p.shape[0]} rays in flight, skipping splits")

        f     = sdf(p).view(-1)
        delta = torch.abs(f)
        near_surface = delta < eps

        # Accept splits only if the new rays are not near the surface to avoid double counting hits
        active = True
        if split:
            active = torch.full_like(t, True, dtype=torch.bool)
            accept = ~near_surface[n_base:]
            active[n_base:] = accept # Deactivate lanes of near-surface splits
            max_t[split_idx[accept]] = t_split[accept]

        # Handle any rays that are newly entering the epsilon band
        entering = ~crossing & near_surface & active
        entering_idx = torch.nonzero(entering, as_tuple=False).squeeze(-1)
        if entering_idx.numel() > 0:
            hit_chunks.append(p[entering_idx])

        # Clear crossing state of exiting rays, and mark newly entering rays as crossing
        crossing = near_surface

        # Take a step 
        step = torch.where(~crossing, delta / step_bound, torch.maximum(delta, eps_half))
        p = p + step[:, None] * d
        t = t + step

        # Stream compaction: keep only active rays
        active &= (t < max_t) & (iters < max_iters)
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        p = p[active_idx]
        d = d[active_idx]
        t = t[active_idx]
        max_t = max_t[active_idx]
        crossing = crossing[active_idx]
        iters = iters[active_idx] # TODO: only advance non-crossing rays (?)
        iters[~crossing] += 1

    p_hits = torch.cat(hit_chunks, dim=0) if len(hit_chunks) > 0 else torch.zeros((0, ray_origin.shape[-1]), dtype=dtype, device=device)
    return p_hits

@torch.no_grad()
def sample_uniform_points(sdf: Callable[[torch.Tensor], torch.Tensor], 
                          num_rays: int, dim: int = 3, 
                          eps: float = 1e-4, step_bound: float = 10,
                          dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None): 
    """ Uniformely sample points on an implicit surface represented by a signed distance function.

    Notes
    -----
    This is an optimized implementation of the method described in the paper "Uniform Sampling of Surfaces by Casting Rays" by Ling et al. 
    It distributes work on the GPU more effectively than the reference implementation, resulting in substantially faster sampling.
    
    Additionally, instead of using rejection sampling to obtain uniformely distributed rays (as described in the original paper),
    this implementation directly samples the projected faces of the [-1, 1]^n cube to obtain valid rays. This is both simpler and faster.

    Parameters
    ----------

    sdf: Callable[[torch.Tensor], torch.Tensor]
        Signed distance function of the implicit surface. Given an array of positions (N,dim), this function should return an array of SDF values (N,) or (N,1).
    num_rays: int
        The number of rays used to generate the sample points. The number of samples is only indirectly influenced by this value as each ray-surface intersection is a valid sample.
    dim: int, optional
        The dimensionality of the space, either 2 or 3 (default is 3).
    eps: float, optional
        Threshold determining the epsilon band around the surface for which ray intersections are computed (default is 1e-4).
    step_bound: float, optional
        Inverse scaling applied to the SDF values when computing the sphere tracing step (default is 10).
        The step size at a point `x` is computed as `sdf(x) / step_bound`.
        This underestimation is particularly useful when the underlying function is only an approximate SDF (e.g., because it is represented by a neural network).
    dtype : Optional[torch.dtype]
        The data type of the returned sample points (uses torch default if not given).
        Since the function performs internal computations with this data type, it also has to match that expected and returned by the `sdf`.
    device : Optional[torch.device]
        The device of the returned sample points (uses torch default if not given).
        Since the function performs internal computations on this device, it also has to match that expected and returned by the `sdf`.
        
    Returns
    -------
    samples: torch.Tensor
        Uniformely distributed sample points on the zero level set, as an array of shape (num_points,dim).
    """

    ray_origin, ray_direction, max_t = sample_uniform_rays(num_rays, dim=dim, dtype=dtype, device=device)     
    return sphere_trace_all_intersections(sdf, ray_origin, ray_direction, max_t, eps=eps, step_bound=step_bound) 

# Class wrapper around the function `sample_uniform_points`, providing an interface that matches the reference implementation by Ling et al.
class ImplicitUniformSampler():
    def __init__(self, thresh: float = 1e-4, dim: int = 3, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.thresh = thresh
        self.dim    = dim
        self.device = device
        self.dtype = dtype

    def sample(self, sdf_func: Callable[[torch.Tensor], torch.Tensor], num_rays: int):
        dtype  = self.dtype
        device = self.device
        # Try to infer the dtype and device using the module parameters (if not given)
        if dtype is None or device is None:
            if isinstance(sdf_func, torch.nn.Module):
                param = next(sdf_func.parameters())
                if dtype is None:
                    dtype  = param.dtype
                if device is None:
                    device = param.device
            else:
                raise RuntimeError("""Unable to infer dtype from the SDF function (it is not a torch.nn.Module).
                                      Please provide device and dtype to the sampler's constructor.""")

        return sample_uniform_points(sdf_func, num_rays, dim=self.dim, eps=self.thresh, device=self.device)