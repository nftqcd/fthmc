# fthmc: Field Transformation HMC

Flowed HMC for Lattice Gauge Theory

- [📕 **HMC with Normalizing Flows** (arXiv:2112.01586)](https://arxiv.org/abs/2112.01586)
- [📓 **Example Notebook**](fthmc/notebooks/fthmc.ipynb)

### Abstract 
We propose using Normalizing Flows as a trainable kernel within the molecular dynamics update of Hamiltonian Monte Carlo (HMC). 
By learning (invertible) transformations that simplify our dynamics, we can outperform traditional methods at generating independent configurations. 
We show that, using a carefully constructed network architecture, our approach can be easily scaled to large lattice volumes with minimal retraining effort. 

The source code for our implementation is publicly available online at [github.com/nftqcd/fthmc](https://www.github.com/nftqcd/fthmc).

### Citation

**Contact**: [Sam Foreman](https://www.samforeman.me)

If you use this code or found this work interesting, please cite our work along with the original paper:
```bibtex
@ARTICLE{2021arXiv211201586F,
       author = {{Foreman}, Sam and {Izubuchi}, Taku and {Jin}, Luchang and {Jin}, Xiao-Yong and {Osborn}, James C. and {Tomiya}, Akio},
        title = "{HMC with Normalizing Flows}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, High Energy Physics - Lattice},
         year = 2021,
        month = dec,
          eid = {arXiv:2112.01586},
        pages = {arXiv:2112.01586},
archivePrefix = {arXiv},
       eprint = {2112.01586},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv211201586F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


## Acknowledgement
> This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under contract DE_AC02-06CH11357. 
>
> This work describes objective technical results and analysis.
>
> Any subjective views or opinions that might be expressed in the work do not necessarily represent the views of the U.S. DOE or the United States Government. Declaration of Interests - None.
