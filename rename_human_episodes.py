#!/usr/bin/env python3
"""Rename sparse human_episode_XXXX files to contiguous ACT episode_N names."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET_DIR = Path(
    "/mnt/additional/Data/DexMimic_Data/Human_data/Peach_in_bowl"
)
SOURCE_PATTERN = re.compile(r"^human_episode_(\d+)\.hdf5$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mapping_name", default="episode_rename_mapping.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    sources = []
    for path in dataset_dir.glob("human_episode_*.hdf5"):
        match = SOURCE_PATTERN.match(path.name)
        if match:
            sources.append((int(match.group(1)), path))
    sources.sort()
    if not sources:
        raise RuntimeError(f"no human_episode_*.hdf5 files found in {dataset_dir}")

    targets = [dataset_dir / f"episode_{index}.hdf5" for index in range(len(sources))]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing ACT episode files: " + ", ".join(existing[:5])
        )

    records = []
    for new_id, ((old_id, source), target) in enumerate(zip(sources, targets)):
        records.append({
            "old_episode_id": old_id,
            "new_episode_id": new_id,
            "old_filename": source.name,
            "new_filename": target.name,
        })

    manifest = {
        "dataset_dir": str(dataset_dir),
        "renamed_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(records),
        "operation": "rename human_episode_<sparse id>.hdf5 to episode_<contiguous id>.hdf5",
        "records": records,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    # The prefixes differ, so every destination was proven absent above and no
    # temporary intermediate names are required.
    for record in records:
        (dataset_dir / record["old_filename"]).rename(
            dataset_dir / record["new_filename"]
        )
    mapping_path = dataset_dir / args.mapping_name
    mapping_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(mapping_path)


if __name__ == "__main__":
    main()
