"""Command-line entry point for P3 image fitting and P4 densification."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="P3/P4: fit an image with 2D Gaussian splats")
    parser.add_argument("image", type=Path, help="Target image, e.g. textures/coffee.png")
    parser.add_argument("--num-gaussians", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-size", type=int, default=128, help="Longest target side in pixels")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--densify", action="store_true", help="Enable P4 cloning, splitting, and pruning")
    parser.add_argument("--budget", type=int, default=1024, help="Maximum count for a densified fit")
    parser.add_argument("--densify-every", type=int, default=200)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--size-threshold", type=float, default=0.02, help="Fraction of image width")
    parser.add_argument("--split-scale", type=float, default=1.6)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--compare-densification", action="store_true",
                        help="Run densified and fixed-count fits at the same actual final count")
    args = parser.parse_args()
    if min(args.num_gaussians, args.steps, args.max_size, args.log_every) <= 0 or args.lr <= 0:
        parser.error("Gaussian count, steps, max-size, log-every, and learning rate must be positive")
    enabled = args.densify or args.compare_densification
    if enabled and (args.budget < args.num_gaussians or args.densify_every <= 0
                    or args.grad_threshold < 0 or args.size_threshold <= 0
                    or args.split_scale <= 1 or not 0 <= args.prune_opacity < 1):
        parser.error("Invalid densification settings; budget must cover the starting count")
    return args


def main():
    args = parse_args()
    from experiments import run_experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
