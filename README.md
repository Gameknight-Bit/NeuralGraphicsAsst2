# Gaussian Splatting Implementation

| File | Responsibility |
|---|---|
| `main.py` | Command-line options and entry point |
| `gaussians.py` | Covariance matrices and Gaussian weights |
| `renderer.py` | Pixel grid and differentiable compositing |
| `fitting.py` | Initialization, Adam training, and final rendering |
| `densification.py` | Cloning, splitting, pruning, and Adam state resizing |
| `results.py` | Count sweep, CSV table, and PSNR plot |

| `experiments.py` | Single-image runs, saved outputs, and fixed-count comparison (from P4) |
| `image_utils.py` | Device selection and image loading/saving |
| `scene_data.py` | Load posed images and the shared camera intrinsics |
| `make_gif.py` | Make a gif of the scene (with the same camera path) for either P7 or P8 |

| `gaussians_3d.py` | Quaternion rotations, 3D covariance, and camera projection |
| `fitting_3d.py` | Initialize, project, render, and fit a 3D scene | 
| `train_3d.py` | Command line, training-view evaluation, and writeup outputs |
| `densification_3d.py` | 3D cloning, splitting, and pruning |
| `evaluation_3d.py` | Per-view metrics and mean PSNR on any camera split |
| `compare_3d.py` | Plain-vs-densified training and held-out comparison |

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Fit one image:

```bash
python src/main.py textures/coffee.png --num-gaussians 256 --steps 2000 --output-dir outputs/p3
```

Run densification and compare with a fixed fit at the same final count:

```bash
python src/main.py textures/coffee.png --num-gaussians 64 --budget 256 --compare-densification --output-dir outputs/p4
```

Run all nine fixed-count fits:
```bash
python src/results.py textures/coffee.png textures/astronaut.png textures/cat.png --counts 256 1024 4096 --steps 2000 --max-size 128 --seed 0 --output-dir outputs/p5
```

Train on 3D spheres scene
```bash
python src/train_3d.py spheres --num-gaussians 1024 --iters 1500 --lr 0.01 --pixel-chunk 4096 --seed 0 --device cuda --output-dir outputs/p7_1024
python src/train_3d.py spheres --num-gaussians 2048 --iters 1500 --lr 0.01 --pixel-chunk 4096 --seed 0 --device cuda --output-dir outputs/p7_2048
```

Train Densification of 3D spheres scene
```bash
```
