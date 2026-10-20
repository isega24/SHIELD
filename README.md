# SHIELD

Official implementation of **SHIELD** (Selective Hidden Input Evaluation for Learning Dynamics) and **X-SHIELD**, regularizations that use explainability to improve both the model and its explanations during training.

The idea is simple: instead of only *measuring* explanations after the fact, hide part of the input during training so the model is forced to generalize without it. The model cannot lean on a handful of dominant features, so it becomes both more accurate and more explainable — and no human-labelled explanations are needed at any point.

Two variants are provided:

| | |
|---|---|
| `shield` | **T-SHIELD.** Hides a random subset of the segmented input at every step. Model-agnostic, no explanation required. |
| `xshield` | **X-SHIELD.** Uses the model's own explanation to choose *which* regions to hide: the least important ones. Ties the regularization to the explanation being produced. |

## Installation

```bash
git clone https://github.com/isega24/SHIELD.git
cd SHIELD
pip install -e .
```

Evaluating explanations (the `tests/` pipeline) also needs the REVEL framework, now maintained as [S-ReVel](https://github.com/ari-dasci/S-ReVel):

```bash
git clone https://github.com/ari-dasci/S-ReVel.git
pip install ./S-ReVel
```

## Usage

Both functions return a scalar score you add to your loss as a regularization term:

```python
import torch
from SHIELD import shield, xshield

model = MyModel().cuda()
x = torch.rand((1, 3, 224, 224)).cuda()

loss = criterion(model(x), y) + lam * xshield(model, x, segmentation=1, device="cuda", percentage=2)
loss.backward()
```

Parameters:

- `model` — the model being trained or evaluated.
- `input` — input tensor.
- `input_0` — the `null` input used to replace hidden regions (same shape as `input`).
- `segmentation` — granularity of the input partition.
- `device` — `cpu` or `cuda`.
- `percentage` — percentage of the input to be hidden.

## Citation

```bibtex
@article{sevillano2026xshield,
  title   = {X-SHIELD: Regularization for explainable Artificial Intelligence},
  author  = {Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal = {Machine Intelligence Research},
  year    = {2026},
  doi     = {10.1007/s11633-025-1576-y}
}
```

## License

GPL-3.0. See [LICENSE](LICENSE).
