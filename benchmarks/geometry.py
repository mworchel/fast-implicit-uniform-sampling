import torch
from typing import Optional, Tuple

torch.no_grad()
def compute_face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = vertices[faces].unbind(1)
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=1)
    return areas

@torch.no_grad()
def _project_points_to_mesh(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project a (chunk of) query points onto the triangles (a, b, c).
    See `project_points_to_mesh` for details. This forms an intermediate tensor of shape
    (num_points, num_faces, 3), which is why the public function processes points in chunks.
    """
    # Triangle vertices, broadcast against the query points: (1, num_faces, 3) vs (num_points, 1, 3)
    a = a.unsqueeze(0)  # (1, M, 3)
    b = b.unsqueeze(0)
    c = c.unsqueeze(0)
    p = points.unsqueeze(1)  # (N, 1, 3)

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)

    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    # Signed areas identifying the barycentric region of the projection
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # Interior of the triangle (default case)
    denom = 1.0 / (va + vb + vc)
    v_bary = (vb * denom).unsqueeze(-1)
    w_bary = (vc * denom).unsqueeze(-1)
    closest = a + v_bary * ab + w_bary * ac  # (N, M, 3)

    # The seven regions are mutually exclusive, so the order of the assignments does not matter.
    def where(mask, value):
        return torch.where(mask.unsqueeze(-1), value, closest)

    # Vertex regions
    closest = where((d1 <= 0) & (d2 <= 0), a.expand_as(closest))   # vertex A
    closest = where((d3 >= 0) & (d4 <= d3), b.expand_as(closest))  # vertex B
    closest = where((d6 >= 0) & (d5 <= d6), c.expand_as(closest))  # vertex C

    # Edge regions
    closest = where((vc <= 0) & (d1 >= 0) & (d3 <= 0),
                    a + (d1 / (d1 - d3)).unsqueeze(-1) * ab)  # edge AB
    closest = where((vb <= 0) & (d2 >= 0) & (d6 <= 0),
                    a + (d2 / (d2 - d6)).unsqueeze(-1) * ac)  # edge AC
    closest = where((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                    b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))).unsqueeze(-1) * (c - b))  # edge BC

    # Keep the nearest triangle for each query point
    sq_dist = ((closest - p) ** 2).sum(-1)  # (N, M)
    face_idx = sq_dist.argmin(dim=1)        # (N,)
    idx = torch.arange(points.shape[0], device=points.device)
    projected = closest[idx, face_idx]
    distances = sq_dist[idx, face_idx].sqrt()

    return projected, distances, face_idx

@torch.no_grad()
def project_points_to_mesh(points: torch.Tensor, v: torch.Tensor, f: torch.Tensor, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project points in R^3 onto the surface of a triangle mesh.

    For each query point, the closest point on the mesh is found by computing the closest
    point on every triangle (fully vectorized) and keeping the nearest one. The per-triangle
    closest point uses the barycentric region test from Ericson, "Real-Time Collision Detection".

    The projection forms an intermediate tensor of shape (chunk_size, num_faces, 3), so its
    memory cost scales as O(chunk_size * num_faces). Use `chunk_size` to bound this cost for
    large inputs; it only affects memory usage, not the result.

    Parameters
    ----------
    points : torch.Tensor
        Query points, as an array of shape (num_points, 3).
    v : torch.Tensor
        Mesh vertices, as an array of shape (num_vertices, 3).
    f : torch.Tensor
        Mesh faces (vertex indices), as an integer array of shape (num_faces, 3).
    chunk_size : Optional[int]
        Number of query points projected at once. If not given, all points are projected in a
        single batch.

    Returns
    -------
    closest : torch.Tensor
        Projected points on the mesh surface, as an array of shape (num_points, 3).
    distances : torch.Tensor
        Distance from each query point to its projection, as an array of shape (num_points,).
    face_idx : torch.Tensor
        Index of the face each point was projected onto, as an array of shape (num_points,).
    """
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]  # (M, 3) each

    if chunk_size is None:
        chunk_size = points.shape[0]

    projected, distances, face_idx = [], [], []
    for chunk in points.split(max(chunk_size, 1)):
        proj, dist, idx = _project_points_to_mesh(chunk, a, b, c)
        projected.append(proj)
        distances.append(dist)
        face_idx.append(idx)

    return torch.cat(projected), torch.cat(distances), torch.cat(face_idx)