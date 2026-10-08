"""P7 runner: fit the spheres training views and save the writeup results."""

import argparse
import csv
import math
from pathlib import Path

import torch

from fitting_3d import fit_scene, render_view
from image_utils import get_device, save_image
from scene_data import load_cameras


def main():
    parser = argparse.ArgumentParser(description="P7: fit 3D Gaussians to posed training views")
    parser.add_argument("dataset", type=Path, help="Extracted folder containing cameras.json and train/")
    parser.add_argument("--num-gaussians", type=int, default=4000)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pixel-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p7"))
    args = parser.parse_args()
    if min(args.num_gaussians, args.iters, args.pixel_chunk) <= 0 or args.lr <= 0:
        parser.error("Gaussian count, iterations, pixel batch size, and lr must be positive")
    torch.manual_seed(args.seed)
    device = args.device or get_device()
    cameras = load_cameras(args.dataset, device=device, split="train")
    print(f"Device: {device}; training views: {len(cameras)}; Gaussians: {args.num_gaussians}")
    parameters, history = fit_scene(cameras, args.num_gaussians, args.iters, args.lr,
                                    args.pixel_chunk, camera_seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "parameters": {name: value.cpu() for name, value in parameters.items()},
        "history": history, "num_gaussians": args.num_gaussians, "iters": args.iters,
        "lr": args.lr, "seed": args.seed, "dataset": str(args.dataset),
    }, args.output_dir / "scene.pt")

    # Evaluate the final parameters on every training view, not just the last one.
    rows, comparisons = [], []
    selected = {0, len(cameras) // 2, len(cameras) - 1}
    with torch.no_grad():
        for i, camera in enumerate(cameras):
            image = render_view(parameters, camera, args.pixel_chunk)
            mse = ((image - camera["image"]) ** 2).mean().item()
            psnr = -10 * math.log10(mse) if mse > 0 else math.inf
            rows.append(dict(view=camera["file"], mse=mse, psnr=psnr))
            if i in selected:
                name = f"train_compare_{i:03d}.png"
                save_image(torch.cat((camera["image"], image), dim=1), args.output_dir / name)
                comparisons.append(f"{camera['file']} — {psnr:.2f} dB\n\n![Ground truth left, render right]({name})")
    with (args.output_dir / "train_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["view", "mse", "psnr"])
        writer.writeheader()
        writer.writerows(rows)
    mean_psnr = sum(row["psnr"] for row in rows) / len(rows)
    report = (
        f"# P7 results\n\nMean training-view PSNR: **{mean_psnr:.2f} dB** over {len(rows)} views "
        "(arithmetic mean of the per-view PSNRs, measured on float renders).\n\n"
        f"Used **{args.num_gaussians} Gaussians**, {args.iters} Adam steps, lr={args.lr}, seed={args.seed}. "
        "Centers were uniform in [-1.5, 1.5]^3, scales were 0.08 on each axis, "
        "quaternions were (1, 0, 0, 0), colors started at sigmoid(0)=0.5, and "
        "opacities at sigmoid(-2)≈0.119. Each step sampled one training camera. "
        "Images retained their original resolution and shared K. Projected covariances "
        "use a 1e-4 pixel-variance diagonal regularizer.\n\nGround truth is on the left; render is on the right.\n\n"
    )
    (args.output_dir / "results.md").write_text(report + "\n\n".join(comparisons) + "\n", encoding="utf-8")
    print(f"Mean training PSNR: {mean_psnr:.2f} dB; saved checkpoint, metrics, comparisons, and report in {args.output_dir}")


if __name__ == "__main__":
    main()
