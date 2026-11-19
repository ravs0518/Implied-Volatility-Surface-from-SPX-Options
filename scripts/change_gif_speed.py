#!/usr/bin/env python3
"""Change an existing GIF's playback speed without refitting the surfaces.

Example:
    python3 scripts/change_gif_speed.py \
        results/svi_surface_2023.gif \
        --fps 8 \
        --output results/svi_surface_2023_fast.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    output = args.output or args.input.with_name(f"{args.input.stem}_{args.fps:g}fps.gif")
    duration_ms = max(1, round(1000 / args.fps))

    with Image.open(args.input) as source:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(source)]
        if not frames:
            raise ValueError("The input GIF contains no frames")
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

    print(f"Wrote {output} at {args.fps:g} FPS ({duration_ms} ms/frame)")
    print(f"Frames: {len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
