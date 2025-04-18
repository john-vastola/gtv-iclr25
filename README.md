# Generalization through variance: how noise shapes inductive biases in diffusion models

This repo contains code that reproduces the figures from "[Generalization through variance: how noise shapes inductive biases in diffusion models](https://openreview.net/forum?id=7lUdo8Vuqa)", a paper accepted to ICLR 2025. 

**Abstract:**
> How diffusion models generalize beyond their training set is not known, and is somewhat mysterious given two facts: the optimum of the denoising score matching (DSM) objective usually used to train diffusion models is the score function of the training distribution; and the networks usually used to learn the score function are expressive enough to learn this score to high accuracy. We claim that a certain feature of the DSM objective---the fact that its target is not the training distribution's score, but a noisy quantity only equal to it in expectation---strongly impacts whether and to what extent diffusion models generalize. In this paper, we develop a mathematical theory that partly explains this 'generalization through variance' phenomenon. Our theoretical analysis exploits a physics-inspired path integral approach to compute the distributions typically learned by a few paradigmatic under- and overparameterized diffusion models. We find that the distributions diffusion models effectively learn to sample from resemble their training distributions, but with 'gaps' filled in, and that this inductive bias is due to the covariance structure of the noisy target used during training. We also characterize how this inductive bias interacts with feature-related inductive biases. 

The paper is mostly theoretical and the code involves only toy examples, so only standard Python libraries (NumPy, SciPy, and Matplotlib) are used.

There is one Jupyter notebook per figure:

**1.** `fig1-variance-visualize.ipynb` Contains code for generating Figure 1, "Visualization of proxy score variance ($\text{tr}(\mathbf{C})/[\text{tr}(\mathbf{C}) + \Vert \mathbf{s} \Vert_2^2]$) for four example 2D data distributions".

<p align="center">
<img src="fig1_proxy_score_cov.png" width="600"/></p>

**2.** `fig2-1d-generalize.ipynb` Contains code for generating Figure 2, "Average learned distribution ($N = 100$) for a linear model with Gaussian features trained on different sample draws from a 1D data distribution { -1, 0, 1 }".

<p align="center">
<img src="fig2_gap_filling.png" width="600"/></p>

**3.** `fig3-2d-generalize.ipynb` Contains code for generating Figure 3, "Generalization of a 2D data distribution depends on features used and data orientation".

<p align="center">
<img src="fig3_feat_noise.png" width="400"/></p>

Three files containing utility functions (`general_functions.py`, `linear_functions_1d.py`, and `linear_functions_2d.py`) are in the `functions/` folder. Intended notebook outputs are in the `results/` folder.

## Citation 

```
@inproceedings{
    vastola2025generalization,
    title={Generalization through variance: how noise shapes inductive biases in diffusion models},
    author={John Vastola},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=7lUdo8Vuqa}
}
```
