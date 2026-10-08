"""P8: compare a plain P7 cloud with a smaller cloud grown by densification."""

import argparse
import csv
from pathlib import Path

import torch

from evaluation_3d import evaluate_scene
from fitting_3d import fit_scene
from image_utils import get_device, save_image
from scene_data import load_cameras


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="P8: plain vs. densified 3D reconstruction")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--initial-count", type=int, default=512)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pixel-chunk", type=int, default=2048)
    parser.add_argument("--densify-every", type=int, default=200)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--world-size-threshold", type=float, default=0.05)
    parser.add_argument("--split-scale", type=float, default=1.6)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p8"))
    args = parser.parse_args()
    if min(args.initial_count, args.budget, args.iters, args.pixel_chunk, args.densify_every) <= 0 or args.lr <= 0:
        parser.error("Counts, iterations, batch size, pass interval, and lr must be positive")
    if args.initial_count > args.budget or args.grad_threshold < 0 or args.world_size_threshold <= 0 or args.split_scale <= 1 or not 0 <= args.prune_opacity < 1:
        parser.error("Invalid densification thresholds or starting count above budget")
    device = args.device or get_device()
    train = load_cameras(args.dataset, device, split="train")
    heldout = load_cameras(args.dataset, device, split="val")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted({0, len(heldout) // 2, len(heldout) - 1})
    settings = dict(budget=args.budget, densify_every=args.densify_every,
                    grad_threshold=args.grad_threshold, size_threshold=args.world_size_threshold,
                    split_scale=args.split_scale, prune_opacity=args.prune_opacity)
    summary, pictures, passes = [], {}, []
    for name, count, adaptive in (("plain", args.budget, False), ("densified", args.initial_count, True)):
        torch.manual_seed(args.seed)
        print(f"\n{name}: start N={count}, budget={args.budget}, device={device}")
        parameters, history = fit_scene(
            train, count, args.iters, args.lr, args.pixel_chunk,
            densification=adaptive, camera_seed=args.seed, **settings,
        )
        actual_count = len(parameters["mu3"])
        torch.save({
            "parameters": {key: value.cpu() for key, value in parameters.items()},
            "history": history, "num_gaussians": actual_count, "initial_count": count,
            "iters": args.iters, "lr": args.lr, "seed": args.seed, "dataset": str(args.dataset),
            "densification": adaptive, "settings": settings,
        }, args.output_dir / f"{name}_scene.pt")
        # Held-out views are used only here, after optimization has finished.
        train_rows, train_psnr, _ = evaluate_scene(parameters, train, args.pixel_chunk)
        val_rows, val_psnr, pictures[name] = evaluate_scene(parameters, heldout, args.pixel_chunk, selected)
        write_csv(args.output_dir / f"{name}_train_metrics.csv", train_rows)
        write_csv(args.output_dir / f"{name}_heldout_metrics.csv", val_rows)
        summary.append(dict(run=name, initial_count=count, gaussians=actual_count,
                            train_psnr=train_psnr, heldout_psnr=val_psnr))
        if adaptive:
            passes = [dict(step=row["step"], **row["densification"])
                      for row in history if "densification" in row]
        print(f"{name}: N={actual_count}, training={train_psnr:.2f} dB, held-out={val_psnr:.2f} dB")
        del parameters, history

    write_csv(args.output_dir / "summary.csv", summary)
    if passes:
        write_csv(args.output_dir / "densification.csv", passes)
    panels = []
    for i in selected:
        filename = f"heldout_compare_{i:03d}.png"
        image = torch.cat((heldout[i]["image"].cpu(), pictures["plain"][i], pictures["densified"][i]), dim=1)
        save_image(image, args.output_dir / filename)
        panels.append(f"{heldout[i]['file']}\n\n![Ground truth, plain, densified]({filename})")
    table = ["| Fit | Initial N | Final N | Training PSNR | Held-out PSNR |",
             "|---|---:|---:|---:|---:|"]
    table += [f"| {r['run']} | {r['initial_count']} | {r['gaussians']} | {r['train_psnr']:.2f} dB | {r['heldout_psnr']:.2f} dB |" for r in summary]
    plain_score, dense_score = summary[0]["heldout_psnr"], summary[1]["heldout_psnr"]
    winner = "densified" if dense_score > plain_score else "plain" if plain_score > dense_score else "tie"
    report = (
        f"# P8 results\n\nBoth fits used {args.iters} Adam steps, lr={args.lr}, seed={args.seed}, "
        f"and the same sampled training-camera sequence. The common budget was {args.budget}; "
        "the table reports actual final counts. PSNR is the arithmetic mean over each split, "
        "computed from float renders. Held-out images did not enter the loss.\n\n"
        "Initialization followed P7: centers uniform in [-1.5,1.5]^3, scales 0.08, identity "
        "quaternions, gray colors, and opacity sigmoid(-2).\n\n"
        f"Every {args.densify_every} steps before the final step, mean position-gradient norms "
        f"were compared with {args.grad_threshold}. Size is now measured in world units: "
        f"max scale <= {args.world_size_threshold} is cloned in place; larger selected Gaussians "
        f"are replaced by two children with scales divided by {args.split_scale}. Split offsets "
        "are sampled in 3D, scaled along the parent's axes, rotated by its quaternion, and "
        "applied with opposite signs. Quaternions, colors, and opacities are copied.\n\n"
        f"Pruning removes opacity < {args.prune_opacity}, including when the budget is full; "
        "one primitive is retained if all would be removed. The largest eligible gradient "
        "scores get remaining capacity. As in P4, surviving Adam moments are preserved, new "
        "rows get zero moments, and gradient averages reset after each pass. Adam's tensor "
        "step counter is retained.\n\n"
        f"Completed {len(passes)} passes, pruning {sum(p['pruned'] for p in passes)} primitives in total.\n\n"
    )
    report += "\n".join(table) + f"\n\nHigher held-out PSNR: **{winner}**.\n\n"
    report += "Panels show **ground truth | plain | densified** for the same held-out camera. Inspect the background and silhouettes for floater reduction.\n\n"
    (args.output_dir / "results.md").write_text(report + "\n\n".join(panels) + "\n", encoding="utf-8")
    print(f"\nSaved P8 checkpoints, metrics, held-out comparisons, and results.md in {args.output_dir}")


if __name__ == "__main__":
    main()
