"""P5: run the three images at three fixed Gaussian counts and plot PSNR.

Run: python results.py textures/coffee.png textures/astronaut.png textures/cat.png
Requires torch, numpy, pillow, and matplotlib; keep all project Python files together.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fitting import fit_image
from image_utils import get_device, load_image, save_image


def main():
    parser = argparse.ArgumentParser(description="P5: fixed-count quality sweep")
    parser.add_argument("images", nargs="*", type=Path,
                        default=[Path(f"textures/{name}.png") for name in ("coffee", "astronaut", "cat")])
    parser.add_argument("--counts", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p5"))
    args = parser.parse_args()
    if min(*args.counts, args.steps, args.max_size) <= 0 or args.lr <= 0:
        parser.error("Counts, steps, image size, and learning rate must be positive")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Target image does not exist: {path}")
    if len({path.stem for path in args.images}) != len(args.images):
        parser.error("Target filenames must have different stems")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device, rows = args.device or get_device(), []

    for path in args.images:
        target = load_image(path, max_size=args.max_size, device=device)
        H, W, _ = target.shape
        save_image(target, args.output_dir / f"{path.stem}_target.png")
        for count in sorted(set(args.counts)):
            torch.manual_seed(args.seed)  # Same P3 initialization rule for every run.
            print(f"\n{path.name}: N={count}, {W}x{H}, {device}")
            image, parameters, _ = fit_image(target, count, args.steps, args.lr,
                                            densification=False)
            mse = ((image - target) ** 2).mean().item()  # Float render, before PNG rounding.
            psnr = -10 * math.log10(mse) if mse > 0 else math.inf
            prefix = args.output_dir / f"{path.stem}_{count}"
            save_image(image, prefix.with_name(prefix.name + ".png"))
            torch.save({name: value.cpu() for name, value in parameters.items()}, prefix.with_name(prefix.name + ".pt"))
            rows.append(dict(image=path.stem, N=count, psnr=psnr, mse=mse,
                             values=9 * count, model_KiB=9 * count * 4 / 1024,
                             H=H, W=W, steps=args.steps, lr=args.lr, seed=args.seed))
            # Write after each fit so finished results survive an interrupted sweep.
            with (args.output_dir / "results.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Final PSNR: {psnr:.2f} dB; float32 model: {9 * count * 4 / 1024:.1f} KiB")
            del image, parameters

    fig, ax = plt.subplots(figsize=(7, 4))
    for path in args.images:
        data = [row for row in rows if row["image"] == path.stem]
        ax.plot([row["N"] for row in data], [row["psnr"] for row in data], "o-", label=path.stem)
    ax.set(xlabel="Number of Gaussians (N)", ylabel="PSNR (dB)", title="2D Gaussian fitting: quality vs. count")
    ax.set_xscale("log", base=2)
    ticks = sorted(set(args.counts))
    ax.set_xticks(ticks, labels=[str(count) for count in ticks])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "psnr_vs_count.png", dpi=180)
    plt.close(fig)

    table = ["| Image | N | PSNR (dB) | Float32 model (KiB) |", "|---|---:|---:|---:|"]
    table += [f"| {r['image']} | {r['N']} | {r['psnr']:.2f} | {r['model_KiB']:.1f} |" for r in rows]
    note = ("Compare the saved renders to judge when each image looks right. Expected: "
            "the cup's smooth regions can use broad Gaussians, the portrait also needs smaller "
            "Gaussians for edges and facial detail, and dense fur needs many small Gaussians. "
            "Check this expectation against your measured curves and renders. Model sizes count "
            "nine float32 values per Gaussian and exclude checkpoint metadata.")
    report = f"# P5 results\n\n{args.steps} Adam steps, lr={args.lr}, seed={args.seed}, longest image side <= {args.max_size}px. Densification disabled.\n\n"
    report += "\n".join(table) + "\n\n![PSNR vs. count](psnr_vs_count.png)\n\n" + note + "\n"
    (args.output_dir / "results.md").write_text(report)
    print(f"\nSaved renders, parameters, results.csv, results.md, and psnr_vs_count.png in {args.output_dir}")


if __name__ == "__main__":
    main()
