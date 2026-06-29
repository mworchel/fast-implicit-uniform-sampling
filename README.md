# Fast Uniform Sampling of Implicit Surfaces

```python
from fast_implicit_uniform_sampling import sample_uniform_points
```

## Citation

[TODO: Give context for Djuren et al. 2026 and Ling et al. 2025]
If you find this code or our method useful for your research, please cite the following papers:

```bibtex
@article{Djuren:2026:ImplicitARAP,
    journal = {Computer Graphics Forum},
    title = {{As-Rigid-As-Possible Regularization for Implicit Surfaces}},
    author = {Djuren, Tobias and Worchel, Markus and Finnendahl, Ugo and Alexa, Marc},
    year = {2026},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.70519}
}
```

```bibtex
@article{Ling:2025:RayCastUniformSampling,
    author = {Ling, Selena and Madan, Abhishek and Sharp, Nicholas and Jacobson, Alec},
    title = {Uniform Sampling of Surfaces by Casting Rays},
    journal = {Computer Graphics Forum},
    volume = {44},
    number = {5},
    pages = {e70202},
    doi = {https://doi.org/10.1111/cgf.70202},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70202},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.70202},
    abstract = {Abstract Randomly sampling points on surfaces is an essential operation in geometry processing. This sampling is computationally straightforward on explicit meshes, but it is much more difficult on other shape representations, such as widely-used implicit surfaces. This work studies a simple and general scheme for sampling points on a surface, which is derived from a connection to the intersections of random rays with the surface. Concretely, given a subroutine to cast a ray against a surface and find all intersections, we can use that subroutine to uniformly sample white noise points on the surface. This approach is particularly effective in the context of implicit signed distance functions, where sphere marching allows us to efficiently cast rays and sample points, without needing to extract an intermediate mesh. We analyze the basic method to show that it guarantees uniformity, and find experimentally that it is significantly more efficient than alternative strategies on a variety of representations. Furthermore, we show extensions to blue noise sampling and stratified sampling, and applications to deform neural implicit surfaces as well as moment estimation.},
    year = {2025}
}
```