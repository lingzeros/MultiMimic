#!/usr/bin/env python3
"""Inspect RGB channel order and depth storage in an ACT HDF5 episode."""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _available_image_names(episode):
    group = episode.get("/observations/images")
    return [] if group is None else list(group.keys())


def _available_depth_paths(episode):
    """Find depth datasets in the current and legacy storage layouts."""
    paths = []
    for group_path in ("/observations/images"):
        group = episode.get(group_path)
        if group is None:
            continue
        for name, item in group.items():
            if isinstance(item, h5py.Dataset) and "depth" in name.lower():
                paths.append(f"{group_path}/{name}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read front_RGB from an episode and compare RGB/BGR interpretations."
    )
    parser.add_argument(
        "episode",
        type=Path,
        help="Episode .hdf5 file, e.g. dataset/episode_0.hdf5",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument(
        "--camera", default="front_RGB", help="Camera dataset name (default: front_RGB)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("front_RGB_channel_compare.png"),
        help="Output comparison image",
    )
    parser.add_argument(
        "--depth-camera",
        default="front_depth",
        help=(
            "Depth dataset name or absolute HDF5 path (default: front_depth). "
            "Searches /observations/depth and legacy /observations/images."
        ),
    )
    parser.add_argument(
        "--depth-output",
        type=Path,
        default=Path("front_depth_inspect.png"),
        help="Output depth visualization",
    )
    parser.add_argument(
        "--depth-scan",
        choices=("frame", "all"),
        default="all",
        help="Scan only the selected frame or the complete episode for depth range",
    )
    return parser.parse_args()


def _read_depth_frame(dataset, frame_index):
    """Read raw depth arrays and common HDF5 encoded-image layouts."""
    stored = dataset[frame_index]

    # Normal layouts: [T,H,W], [T,H,W,1], or [T,H,W,3].
    if dataset.ndim in (3, 4):
        depth = np.asarray(stored)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth, False

    # Some datasets save PNG/JPEG bytes as a variable-length [T] dataset or
    # a padded uint8 [T,N] dataset. Decode without changing the bit depth.
    if dataset.ndim in (1, 2):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Depth appears to contain encoded bytes, but OpenCV is unavailable"
            ) from exc

        if isinstance(stored, (bytes, bytearray, np.bytes_)):
            encoded = np.frombuffer(stored, dtype=np.uint8)
        else:
            encoded = np.asarray(stored, dtype=np.uint8).reshape(-1)
        depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(
                f"Could not decode depth frame from dataset shape {dataset.shape}"
            )
        return depth, True

    raise ValueError(f"Unsupported depth dataset shape: {dataset.shape}")


def _describe_depth_format(depth, encoded):
    if depth.ndim == 2:
        if depth.dtype == np.uint16:
            base = "single-channel uint16 raw depth (typical Z16/sensor units)"
        elif np.issubdtype(depth.dtype, np.floating):
            base = "single-channel floating-point metric/scaled depth"
        elif np.issubdtype(depth.dtype, np.integer):
            base = f"single-channel {depth.dtype} integer depth"
        else:
            base = f"single-channel depth with dtype {depth.dtype}"
    elif depth.ndim == 3 and depth.shape[-1] in (3, 4):
        base = (
            f"{depth.shape[-1]}-channel depth visualization/pseudo-color image; "
            "not a raw scalar depth map"
        )
    else:
        base = f"unrecognized decoded depth layout {depth.shape}"
    return f"encoded image bytes -> {base}" if encoded else base


def _depth_counts(depth):
    values = np.asarray(depth)
    if np.issubdtype(values.dtype, np.floating):
        finite = np.isfinite(values)
    else:
        finite = np.ones(values.shape, dtype=bool)
    positive = finite & (values > 0)
    return {
        "total": values.size,
        "finite": int(finite.sum()),
        "zero": int((finite & (values == 0)).sum()),
        "positive": int(positive.sum()),
        "minimum": float(values[finite].min()) if np.any(finite) else np.nan,
        "maximum": float(values[finite].max()) if np.any(finite) else np.nan,
        "positive_minimum": float(values[positive].min()) if np.any(positive) else np.nan,
        "positive_maximum": float(values[positive].max()) if np.any(positive) else np.nan,
    }


def _merge_depth_counts(total, current):
    if total is None:
        return dict(current)
    for key in ("total", "finite", "zero", "positive"):
        total[key] += current[key]
    total["minimum"] = min(total["minimum"], current["minimum"])
    total["maximum"] = max(total["maximum"], current["maximum"])
    if np.isfinite(current["positive_minimum"]):
        total["positive_minimum"] = min(
            total["positive_minimum"], current["positive_minimum"]
        )
        total["positive_maximum"] = max(
            total["positive_maximum"], current["positive_maximum"]
        )
    return total


def _print_depth_stats(label, stats):
    total = max(stats["total"], 1)
    invalid = stats["total"] - stats["finite"]
    print(f"{label} full range:       [{stats['minimum']}, {stats['maximum']}]")
    print(
        f"{label} positive range:   "
        f"[{stats['positive_minimum']}, {stats['positive_maximum']}]"
    )
    print(
        f"{label} zero pixels:      {stats['zero']}/{stats['total']} "
        f"({100.0 * stats['zero'] / total:.3f}%)"
    )
    print(
        f"{label} NaN/Inf pixels:   {invalid}/{stats['total']} "
        f"({100.0 * invalid / total:.3f}%)"
    )


def _save_depth_visualization(depth, output, frame_index):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)

    if depth.ndim == 2:
        values = depth.astype(np.float64)
        valid = np.isfinite(values) & (values > 0)
        if np.any(valid):
            low, high = np.percentile(values[valid], [1, 99])
            if high <= low:
                high = low + 1.0
            shown = np.ma.masked_where(~valid, values)
            image = axis.imshow(shown, cmap="turbo", vmin=low, vmax=high)
            fig.colorbar(image, ax=axis, label="stored depth value")
            axis.set_title(
                f"Depth frame {frame_index} (display clipped to P1={low:.3g}, P99={high:.3g})"
            )
        else:
            axis.imshow(values, cmap="gray")
            axis.set_title(f"Depth frame {frame_index} (no positive valid pixels)")
    else:
        shown = depth[..., :3]
        # OpenCV-decoded 3-channel data is BGR; swapping is only for display.
        axis.imshow(shown[..., ::-1])
        axis.set_title(f"Depth frame {frame_index} (stored pseudo-color image)")

    axis.axis("off")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"depth visualization saved to: {output.resolve()}")


def _inspect_depth(episode, args):
    depth_candidates = _available_depth_paths(episode)
    requested = args.depth_camera
    requested_paths = (
        [requested]
        if requested.startswith("/")
        else [
            f"/observations/depth/{requested}",
            f"/observations/images/{requested}",
        ]
    )
    depth_key = next((path for path in requested_paths if path in episode), None)

    if depth_key is None:
        same_name = [
            path for path in depth_candidates if Path(path).name == Path(requested).name
        ]
        if len(same_name) == 1:
            depth_key = same_name[0]
    if depth_key is None and len(depth_candidates) == 1:
        depth_key = depth_candidates[0]
        print(
            f"requested depth {requested_paths} not found; "
            f"using detected dataset {depth_key}"
        )
    if depth_key is None:
        image_names = _available_image_names(episode)
        print("\n=== Depth inspection ===")
        print(f"No depth dataset found (requested {requested_paths}).")
        print(f"Available depth datasets: {depth_candidates or '(none)'}")
        print(f"Available image datasets: {image_names or '(none)'}")
        if not depth_candidates:
            print("Conclusion: this episode does not contain a stored depth map.")
        else:
            print(
                "Multiple depth candidates found; select one by full path with "
                f"--depth-camera: {depth_candidates}"
            )
        return

    dataset = episode[depth_key]
    if not -dataset.shape[0] <= args.frame < dataset.shape[0]:
        raise IndexError(
            f"Depth frame {args.frame} is outside valid range "
            f"[-{dataset.shape[0]}, {dataset.shape[0] - 1}]"
        )

    frame, encoded = _read_depth_frame(dataset, args.frame)
    attrs = {key: dataset.attrs[key] for key in dataset.attrs}
    frame_stats = _depth_counts(frame)

    print("\n=== Depth inspection ===")
    print(f"dataset:             {depth_key}")
    print(f"stored shape:        {dataset.shape}")
    print(f"stored dtype:        {dataset.dtype}")
    print(f"compression:         {dataset.compression or '(none)'}")
    print(f"chunks:              {dataset.chunks or '(contiguous)'}")
    print(f"attrs:               {attrs or '(none)'}")
    print(f"decoded frame shape: {frame.shape}")
    print(f"decoded frame dtype: {frame.dtype}")
    print(f"format assessment:   {_describe_depth_format(frame, encoded)}")
    _print_depth_stats("selected frame", frame_stats)

    if frame.ndim == 2:
        valid = np.isfinite(frame) & (frame > 0)
        if np.any(valid):
            percentiles = np.percentile(frame[valid], [1, 5, 50, 95, 99])
            print(
                "selected frame positive percentiles P1/P5/P50/P95/P99: "
                f"{percentiles}"
            )

    if args.depth_scan == "all":
        episode_stats = None
        for index in range(dataset.shape[0]):
            current, _ = _read_depth_frame(dataset, index)
            episode_stats = _merge_depth_counts(episode_stats, _depth_counts(current))
        _print_depth_stats("complete episode", episode_stats)

    unit_attrs = {
        str(key).lower(): value
        for key, value in {**dict(episode.attrs), **attrs}.items()
    }
    unit_keys = [
        key for key in unit_attrs
        if "unit" in key or "scale" in key or "depth" in key
    ]
    if unit_keys:
        print("depth-related metadata:")
        for key in unit_keys:
            print(f"  {key}: {unit_attrs[key]}")
    elif frame.dtype == np.uint16 and frame.ndim == 2:
        print(
            "unit assessment: uint16 values are raw integer depth values. "
            "RealSense Z16 commonly needs the sensor depth_scale (often 0.001 m), "
            "but this file has no unit/scale metadata, so the unit cannot be proven from HDF5 alone."
        )
    elif np.issubdtype(frame.dtype, np.floating) and frame.ndim == 2:
        print(
            "unit assessment: floating-point depth may be meters or normalized values; "
            "the exact unit cannot be proven because no unit/scale metadata is stored."
        )

    _save_depth_visualization(frame, args.depth_output, args.frame)


def main():
    args = parse_args()
    dataset_key = f"/observations/images/{args.camera}"

    with h5py.File(args.episode, "r") as episode:
        image_names = _available_image_names(episode)
        if dataset_key not in episode:
            raise KeyError(f"Missing {dataset_key}; available cameras: {image_names}")

        dataset = episode[dataset_key]
        if dataset.ndim != 4 or dataset.shape[-1] not in (3, 4):
            raise ValueError(
                f"Expected [T, H, W, C] with 3/4 channels, got {dataset.shape}"
            )
        if not -dataset.shape[0] <= args.frame < dataset.shape[0]:
            raise IndexError(
                f"Frame {args.frame} is outside valid range "
                f"[-{dataset.shape[0]}, {dataset.shape[0] - 1}]"
            )

        image = np.asarray(dataset[args.frame])[..., :3]
        attrs = {key: dataset.attrs[key] for key in dataset.attrs}
        full_shape = dataset.shape
        stored_dtype = dataset.dtype

        # Depth is inspected while the HDF5 file is open. Missing depth is
        # reported but does not prevent the existing RGB inspection.
        _inspect_depth(episode, args)

    print(f"file:       {args.episode}")
    print(f"dataset:    {dataset_key}")
    print(f"full shape: {full_shape}")
    print(f"dtype:      {stored_dtype}")
    print(f"attrs:      {attrs or '(none)'}")
    print(f"frame:      {args.frame}, range=[{image.min()}, {image.max()}]")
    print(f"channel mean (stored C0/C1/C2): {image.mean(axis=(0, 1))}")

    # Matplotlib expects RGB. The right panel swaps stored channels 0 and 2,
    # showing what the same bytes look like if the HDF5 array is actually BGR.
    rgb_view = image
    bgr_view = image[..., ::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(rgb_view)
    axes[0].set_title("Stored array interpreted as RGB")
    axes[1].imshow(bgr_view)
    axes[1].set_title("Stored array interpreted as BGR")
    for axis in axes:
        axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"comparison saved to: {args.output.resolve()}")
    print("The panel with natural red/blue colors indicates the likely channel order.")


if __name__ == "__main__":
    main()
