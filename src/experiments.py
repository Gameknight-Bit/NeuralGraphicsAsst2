"""Single-image experiments, checkpoint export, and the P4 comparison."""

import json
import math

import torch

from fitting import fit_image
from image_utils import get_device, load_image, save_image


def save_fit(target, image, parameters, history, prefix, args, densification):
    """Save the float-render metrics, PNG, and learned raw parameters."""
    H, W, _ = target.shape
    mse = ((image - target) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse) if mse > 0 else math.inf
    count = len(parameters["mu"])
    print(f"Final N={count}, MSE={mse:.6f}, PSNR={psnr:.2f} dB")
    render_path = prefix.with_name(prefix.name + "_fit.png")
    save_image(image, render_path)
    checkpoint_path = prefix.with_name(prefix.name + "_gaussians.pt")
    torch.save({
        "parameters": {name: value.cpu() for name, value in parameters.items()},
        "H": H, "W": W, "order": list(range(count)),
        "mse": mse, "psnr": psnr, "history": history,
        "steps": args.steps, "lr": args.lr, "seed": args.seed,
        "densification": densification,
    }, checkpoint_path)
    return {"count": count, "mse": mse, "psnr": psnr, "render": str(render_path)}



def run_experiment(args):
    """Run a P3/P4 fit, save outputs, and optionally compare equal counts."""
    enabled = args.densify or args.compare_densification
    densification_options = {
        "densification": enabled, "budget": args.budget,
        "densify_every": args.densify_every, "grad_threshold": args.grad_threshold,
        "size_threshold": args.size_threshold, "split_scale": args.split_scale,
        "prune_opacity": args.prune_opacity,
    }

    torch.manual_seed(args.seed)
    device = args.device or get_device()
    target = load_image(args.image, max_size=args.max_size, device=device)
    H, W, _ = target.shape
    print(f"Device: {device}; target: {W}x{H}; Gaussians: {args.num_gaussians}")
    image, parameters, history = fit_image(
        target, num_gaussians=args.num_gaussians, steps=args.steps,
        lr=args.lr, log_every=args.log_every,
        **densification_options,
    )

    prefix = args.output_dir / args.image.stem
    save_image(target, prefix.with_name(prefix.name + "_target.png"))
    run_prefix = prefix.with_name(prefix.name + "_densified") if enabled else prefix
    result = save_fit(target, image, parameters, history, run_prefix, args, densification_options)

    if args.compare_densification:
        # Match the actual final count, even when growth stops below the budget.
        final_count = result["count"]
        del image, parameters, history
        torch.manual_seed(args.seed)
        print(f"\nFixed-count comparison: {final_count} Gaussians for {args.steps} steps")
        fixed_image, fixed_parameters, fixed_history = fit_image(
            target, num_gaussians=final_count, steps=args.steps,
            lr=args.lr, log_every=args.log_every,
        )
        fixed = save_fit(target, fixed_image, fixed_parameters, fixed_history,
                         prefix.with_name(prefix.name + "_fixed"), args, {"densification": False})
        winner = ("densified" if result["psnr"] > fixed["psnr"] else
                  "fixed" if fixed["psnr"] > result["psnr"] else "tie")
        delta = 0.0 if winner == "tie" else result["psnr"] - fixed["psnr"]
        comparison = {
            "target": str(args.image), "H": H, "W": W, "seed": args.seed,
            "steps_per_run": args.steps, "lr": args.lr,
            "starting_count": args.num_gaussians, "budget": args.budget,
            "densified": result, "fixed": fixed, "higher_psnr": winner,
            "densified_minus_fixed_db": delta, "rule": densification_options,
        }
        comparison_path = prefix.with_name(prefix.name + "_comparison.json")
        # JSON represents a perfect reconstruction's infinite PSNR as a string.
        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            return value
        comparison_path.write_text(json.dumps(json_safe(comparison), indent=2) + "\n")
        print(f"Comparison: {winner}; densified minus fixed PSNR = {delta:+.2f} dB")
        print(f"Saved {comparison_path}")
    print(f"Saved reconstruction, resized target, and Gaussian checkpoint in {args.output_dir}")

