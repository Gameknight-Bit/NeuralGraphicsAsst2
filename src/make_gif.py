"""Evaluate a saved scene and render a short novel camera arc; no training."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def arc_pose(R, t, degrees):
    """Rotate a camera around the world origin about its original up axis.

    R is world-to-camera. Rotating both its center and orientation by Q gives
    R_new = R @ Q.T and t_new = t. This preserves distance and framing of origin.
    """
    axis = -R[1] / np.linalg.norm(R[1])
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.deg2rad(degrees)
    Q = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return R @ Q.T, t.copy()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--arc-degrees', type=float, default=40,
                        help='Total arc centered on the first held-out camera (default: 40)')
    parser.add_argument('--pixel-chunk', type=int, default=4096)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/p9'))
    args = parser.parse_args()
    if args.frames < 2 or args.pixel_chunk < 1 or not 0 < args.arc_degrees <= 360:
        parser.error('Use at least 2 frames, a positive pixel chunk, and an arc in (0, 360]')

    import torch
    from evaluation_3d import evaluate_scene
    from fitting_3d import render_view
    from image_utils import get_device, save_image
    from scene_data import load_cameras

    device = args.device or get_device()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    parameters = {key: value.to(device) for key, value in checkpoint['parameters'].items()}
    count = len(parameters['mu3'])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    with torch.no_grad():
        for split in ('train', 'val'):
            cameras = load_cameras(args.dataset, device, split=split)
            selected = sorted({0, len(cameras) // 2, len(cameras) - 1})
            rows, mean_psnr, renders = evaluate_scene(parameters, cameras, args.pixel_chunk, selected)
            write_csv(out / f'{split}_metrics.csv', rows)
            for i in selected:
                panel = torch.cat((cameras[i]['image'].cpu(), renders[i]), dim=1)
                save_image(panel, out / f'{split}_compare_{i:03d}.png')
            summary.append(dict(split=split, views=len(cameras), gaussians=count, psnr=mean_psnr))
            print(f'{split}: mean PSNR={mean_psnr:.2f} dB, N={count}', flush=True)

        # The last loaded split is held-out. Keep its intrinsics and image size.
        base = cameras[0]
        R, t = base['R_wc'].cpu().numpy(), base['t'].cpu().numpy()
        angles = np.linspace(-args.arc_degrees / 2, args.arc_degrees / 2, args.frames)
        frames, poses = [], []
        for i, angle in enumerate(angles):
            rotation, translation = arc_pose(R, t, angle)
            camera = {key: base[key] for key in ('K', 'H', 'W')}
            camera.update(R_wc=base['R_wc'].new_tensor(rotation), t=base['t'].new_tensor(translation))
            filename = f'novel_{i:03d}.png'
            save_image(render_view(parameters, camera, args.pixel_chunk), out / filename)
            with Image.open(out / filename) as image:
                frames.append(image.convert('RGB'))
            poses.append(dict(file=filename, angle_degrees=float(angle),
                              R_wc=rotation.tolist(), t=translation.tolist()))
        # Ping-pong playback avoids a jump between the endpoints of the arc.
        playback = frames + frames[-2:0:-1]
        playback[0].save(out / 'novel_arc.gif', save_all=True,
                         append_images=playback[1:], duration=100, loop=0)

    write_csv(out / 'summary.csv', summary)
    pose_data = dict(K=base['K'].cpu().tolist(), width=base['W'], height=base['H'], frames=poses)
    (out / 'novel_cameras.json').write_text(json.dumps(pose_data, indent=2), encoding='utf-8')
    train_psnr, val_psnr = (row['psnr'] for row in summary)
    report = (
        f'# P9 results\n\nCheckpoint: {args.checkpoint}\n\nGaussians: {count}\n\n'
        f'Mean training PSNR: {train_psnr:.2f} dB\n\n'
        f'Mean held-out PSNR: {val_psnr:.2f} dB\n\n'
        f'Training minus held-out: {train_psnr - val_psnr:.2f} dB\n\n'
        'Metrics are arithmetic means of per-view PSNRs from float renders, before PNG/GIF conversion. '
        'No optimization was performed. Comparison panels show ground truth left, render right.\n\n'
        f'The novel sequence follows a {args.arc_degrees:g}-degree arc about the world origin, '
        'using the first held-out camera\'s up axis and preserving its intrinsics. '
        'These generated views have no ground truth, so no PSNR is assigned to them.\n\n'
        '![Novel camera arc](novel_arc.gif)\n\n'
        'Inspect silhouettes, background haze, gaps, and changes between frames. '
        'Describe only artifacts visible in your results; distinguish limited coverage, '
        'limited Gaussian capacity, and inconsistent geometry as possible explanations.\n'
    )
    (out / 'results.md').write_text(report, encoding='utf-8')
    print(f'Saved metrics, comparisons, novel PNG frames, and novel_arc.gif in {out}')


if __name__ == '__main__':
    main()
