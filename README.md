# Gaussian Splatting Implementation

| File | Responsibility |
|---|---|
| `main.py` | Command-line options and entry point | x
| `gaussians.py` | P1: covariance matrices and Gaussian weights | x
| `renderer.py` | P2: pixel grid and differentiable compositing | x
| `fitting.py` | P3: initialization, Adam training, and final rendering | x
| `densification.py` | P4: cloning, splitting, pruning, and Adam state resizing | x
| `results.py` | P5: count sweep, CSV table, and PSNR plot | x

| `experiments.py` | Single-image runs, saved outputs, and P4 fixed-count comparison |
| `image_utils.py` | Device selection and image loading/saving |
| `scene_data.py` | Load posed images and the shared camera intrinsics |

| `gaussians_3d.py` | P6: quaternion rotations, 3D covariance, and camera projection | x
| `fitting_3d.py` | P7: initialize, project, render, and fit a 3D scene | 
| `train_3d.py` | P7: command line, training-view evaluation, and writeup outputs |
| `densification_3d.py` | P8: 3D cloning, splitting, and pruning |
| `evaluation_3d.py` | Per-view metrics and mean PSNR on any camera split |
| `compare_3d.py` | P8 plain-vs-densified training and held-out comparison |

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Fit one image (P3):

```bash
python main.py textures/coffee.png --num-gaussians 256 --steps 2000
```

Run densification and compare with a fixed fit at the same final count (P4):

```bash
python main.py textures/coffee.png --num-gaussians 64 --budget 256 --compare-densification
```

Run all nine fixed-count fits (P5):

```bash
python results.py textures/coffee.png textures/astronaut.png textures/cat.png
```

The renderer demo still runs with `python renderer.py`.
Use `python main.py --help` or `python results.py --help` for the existing options.
P3/P4 outputs go to `outputs/`; P5 outputs go to `outputs/p5/` by default.

To call the fitting function from your own script:

```python
from fitting import fit_image
from image_utils import get_device, load_image, save_image

target = load_image("textures/coffee.png", max_size=128, device=get_device())
image, parameters, history = fit_image(target, num_gaussians=256, steps=2000)
save_image(image, "outputs/coffee_fit.png")
```

For P6, use the projected Gaussians with the same P2 renderer. Given scene
tensors (`mu3`, `scale`, `quat`, `color`, `opacity`) and camera tensors
(`R_wc`, `t`, `K`), all on the same device and using the same floating dtype:

```python
from gaussians_3d import covariance_3d, project_gaussian
from renderer import render

visible = (mu3 @ R_wc.T + t)[:, 2] > 1e-3
Sigma3 = covariance_3d(scale[visible], quat[visible])
mu2, Sigma2, depth = project_gaussian(mu3[visible], Sigma3, R_wc, t, K)
image = render(mu2, Sigma2, color[visible], opacity[visible], depth.argsort(), H, W)
```

Scales are positive standard deviations. Quaternions use `(w, x, y, z)` order;
initialize them to `(1, 0, 0, 0)` for identity rotations. Colors and opacities
passed to the renderer must be in `[0, 1]`. Projection is a local linear
approximation, and the returned depth is camera-space z. `H` and `W` must match
the camera intrinsics.

For P7, extract `spheres.zip` and pass the folder that directly contains
`cameras.json`, `train/`, and `val/`:

```bash
python train_3d.py spheres --num-gaussians 4000 --iters 1500
```

Each step samples one training camera. The scene starts with random centers in
`[-1.5, 1.5]^3`, scale `0.08`, identity quaternions, gray colors, and opacity
`sigmoid(-2)`. Camera intrinsics and image resolution stay as provided. The
training loop only uses the `frames` list; held-out `val_frames` are not fitted.

P7 saves these files in `outputs/p7/`:

- `scene.pt`: learned raw parameters, training history, and run settings.
- `train_metrics.csv`: final MSE and PSNR for every training view.
- `train_compare_*.png`: three examples, ground truth on the left and render on the right.
- `results.md`: mean training-view PSNR, initialization details, and comparisons.

The reported mean is the arithmetic mean of per-view PSNRs. Metrics use the
floating-point renders before PNG conversion. A `1e-4` diagonal pixel-variance
regularizer stabilizes the projected covariance solve. All five parameter
groups are optimized; the P7 Gaussian count stays fixed.

For a quick local smoke run:

```bash
python train_3d.py spheres --num-gaussians 32 --iters 3 --output-dir outputs/p7_smoke
```

If you need smaller pixel batches, add `--pixel-chunk 512`. Every pixel still
contributes to the same view loss before the Adam update.

For P8, train a plain cloud at the P7 budget and a smaller adaptive cloud:

```bash
python compare_3d.py spheres --initial-count 512 --budget 4000 --iters 1500
```

The adaptive run averages 3D position-gradient magnitudes across sampled training
cameras. Every 200 steps before the last step, it prunes opacity below `0.005`,
then clones/splits candidates scoring above `2e-4`, prioritizing larger scores
within the remaining budget. The size threshold is `0.05` **world units**:
small Gaussians clone in place; larger ones split with sampled 3D offsets and
scales divided by `1.6`. Quaternions copy to the children. P4's existing Adam
bookkeeping preserves surviving moments and zeros new rows.

Both fits use the same number of updates, learning rate, seed, and independently
seeded training-camera sequence. The plain run starts at the budget; the adaptive
run may finish below it. `outputs/p8/summary.csv` reports the actual final count
and mean training/held-out PSNR for each. Validation frames are evaluated only
after fitting and never enter optimization.

`outputs/p8/` also contains both checkpoints, per-view CSVs, a densification pass
log when passes ran, and `results.md` with the rule and measurements. Each
`heldout_compare_*.png` shows **ground truth | plain | densified** for one held-out
camera. Compare the background and silhouettes to assess floater reduction.
Results determine whether densification improved PSNR; no improvement is assumed.

Tune the rule with `--grad-threshold`, `--world-size-threshold`, `--prune-opacity`,
`--split-scale`, and `--densify-every`. A small smoke run that exercises passes:

```bash
python compare_3d.py spheres --initial-count 16 --budget 32 --iters 3 --densify-every 1 --output-dir outputs/p8_smoke
```

Reusable modules import each other directly; none imports `main.py`.
This refactor preserves the fitting and densification algorithms, defaults,
command-line flags, and output formats. Target images are not included.
