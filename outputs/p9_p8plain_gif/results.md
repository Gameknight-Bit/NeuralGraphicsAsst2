# P9 results

Checkpoint: outputs\p8_1024\plain_scene.pt

Gaussians: 1024

Mean training PSNR: 22.17 dB

Mean held-out PSNR: 17.20 dB

Training minus held-out: 4.97 dB

Metrics are arithmetic means of per-view PSNRs from float renders, before PNG/GIF conversion. No optimization was performed. Comparison panels show ground truth left, render right.

The novel sequence follows a 40-degree arc about the world origin, using the first held-out camera's up axis and preserving its intrinsics. These generated views have no ground truth, so no PSNR is assigned to them.

![Novel camera arc](novel_arc.gif)

Inspect silhouettes, background haze, gaps, and changes between frames. Describe only artifacts visible in your results; distinguish limited coverage, limited Gaussian capacity, and inconsistent geometry as possible explanations.
