#!/usr/bin/env python3
"""Flag generated/terrains/<preset>/<seed> dirs that are stale or orphaned
relative to the current tarantula_terrain code.

"Stale" means metadata.json's generator_schema_version doesn't match (or is
missing, for dirs generated before the field existed) -- the dir's terrain.sdf/
world.sdf may no longer match what generating it today would produce (e.g.
surface friction/contact tuning, surround_copies tiling). "Orphaned" means the
preset name isn't one tarantula_terrain.generate/nav_maze can produce anymore.

Regenerate a stale or orphaned dir with:
    python3 -m tarantula_terrain.generate --preset <preset> --seed <seed>
or, for nav_maze:
    python3 -m tarantula_terrain.nav_maze --seed <seed>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tarantula_terrain.exporters import GENERATOR_SCHEMA_VERSION
from tarantula_terrain.terrain_cfg import PRESETS

KNOWN_PRESETS = set(PRESETS) | {"nav_maze"}


def check(terrains_root: Path) -> tuple[list[str], list[str], list[str]]:
    stale, orphaned, fresh = [], [], []
    for metadata_path in sorted(terrains_root.glob("*/*/metadata.json")):
        rel = metadata_path.parent.relative_to(terrains_root)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        preset = metadata.get("preset")
        if preset not in KNOWN_PRESETS:
            orphaned.append(f"{rel} (preset={preset!r} no longer exists in terrain_cfg.PRESETS/nav_maze)")
            continue
        version = metadata.get("generator_schema_version")
        if version != GENERATOR_SCHEMA_VERSION:
            stale.append(f"{rel} (generator_schema_version={version!r}, current={GENERATOR_SCHEMA_VERSION})")
        else:
            fresh.append(str(rel))
    return stale, orphaned, fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrains-root", default="generated/terrains")
    args = parser.parse_args()

    stale, orphaned, fresh = check(Path(args.terrains_root))
    for rel in fresh:
        print(f"[OK]      {rel}")
    for entry in stale:
        print(f"[STALE]   {entry}")
    for entry in orphaned:
        print(f"[ORPHAN]  {entry}")
    if not (stale or orphaned):
        print("all generated terrain dirs match the current generator schema.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
