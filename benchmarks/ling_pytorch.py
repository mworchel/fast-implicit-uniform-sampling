from typing import Optional
import torch

# PyTorch port of Ling et al.'s `sample_rays` numpy implementation 
def sample_rays(num_rays: int, dim: int = 3, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):           
    # sample line directions 
    dirs = torch.randn(num_rays, dim, dtype=dtype, device=device)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.unsqueeze(1) #[n,1,3]

    # find the other normal and binormal basis 
    _,_,V = torch.linalg.svd(dirs) 
    E = V[:,:,1:] #[n,3,2] 
    
    # sample random offsets in normal and binormal directions
    dirs = dirs[:,0,:] 
    U = torch.sqrt(torch.tensor(dim, dtype=dtype, device=device)) * ( torch.rand(num_rays, 1, dim-1, dtype=dtype, device=device) * 2.0 - 1.0 )  
    O = torch.sum(U * E, dim=-1) 
    O = O + dirs * torch.sqrt(torch.tensor(dim, dtype=dtype, device=device))
    
    # determine if the ray O+D*t  intersects the [-1,1]^dim hypercube
    # using Slab method
    t_low = torch.zeros((num_rays, dim), dtype=dtype, device=device)
    t_high = torch.zeros((num_rays, dim), dtype=dtype, device=device)
    t_low = (-1.0 - O)/dirs 
    t_high = (1.0 - O)/dirs
    
    t_close = torch.minimum(t_low, t_high)
    t_far = torch.maximum(t_low,t_high)
    t_close = torch.max(t_close, dim=-1)[0]
    t_far = torch.min(t_far,dim=-1)[0]
    t = torch.stack([t_close, t_far],dim=-1)
    keep = t_close < t_far
    
    dirs = dirs[keep]
    O = O[keep]
    t_close = t_close[keep]
    t_far = t_far[keep]
    O = O + dirs * t_close[:,None]
    T = t_far - t_close 
    n_current = O.shape[0]
    
    if (not torch.any(keep)) and n_current > num_rays:
        return torch.zeros((0,3)),torch.zeros((0,3)),torch.zeros((0))
    elif n_current == num_rays:
        return O, dirs, T
    elif n_current < num_rays: 
        O_current, D_current,T_current = sample_rays(num_rays-n_current, dim, dtype, device)
        O = torch.cat([O,O_current],dim=0)
        D = torch.cat([dirs,D_current],dim=0)
        T = torch.cat([T,T_current],dim=0)
        return O, D, T 