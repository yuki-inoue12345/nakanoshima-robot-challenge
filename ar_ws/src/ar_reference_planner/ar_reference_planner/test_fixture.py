"""Generate deterministic map fixtures for hardware-free integration tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from PIL import Image
import yaml


WIDTH = 400
HEIGHT = 400
RESOLUTION = 0.05
ORIGIN = [-5.0, -5.0, 0.0]


def _obstacle_pixels() -> set[tuple[int, int]]:
    """Return cells blocking the direct WP4-to-WP5 test segment."""
    cells = set()
    for row in range(HEIGHT):
        y = ORIGIN[1] + (row + 0.5) * RESOLUTION
        if not 5.35 <= y <= 6.20:
            continue
        for column in range(WIDTH):
            x = ORIGIN[0] + (column + 0.5) * RESOLUTION
            if 2.25 <= x <= 3.35:
                cells.add((column, row))
    return cells


def _write_map(path: Path, occupied: set[tuple[int, int]]) -> Path:
    """Write a trinary PGM/YAML pair matching nav2_map_server semantics."""
    image_path = path.with_suffix('.pgm')
    image = Image.new('L', (WIDTH, HEIGHT), color=255)
    pixels = image.load()
    for column, occupancy_row in occupied:
        image_row = HEIGHT - 1 - occupancy_row
        pixels[column, image_row] = 0
    image.save(image_path)
    metadata = {
        'image': image_path.name,
        'mode': 'trinary',
        'resolution': RESOLUTION,
        'origin': ORIGIN,
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    return path


def create_test_map_bundle(keepout_enabled: bool) -> tuple[Path, Path | None]:
    """Create a selected map and matching optional Keepout mask in ``/tmp``."""
    root = Path(tempfile.mkdtemp(prefix='ar_reference_keepout_test_'))
    selected = _write_map(
        root / 'synthetic.yaml',
        set() if keepout_enabled else _obstacle_pixels(),
    )
    state_dir = root / '.keepout' / selected.stem
    state_dir.mkdir(parents=True)
    settings = {
        'enabled': bool(keepout_enabled),
        'source_map': str(selected),
        'map_metadata': {
            'resolution': RESOLUTION,
            'origin': ORIGIN,
            'size': [WIDTH, HEIGHT],
        },
    }
    (state_dir / 'settings.json').write_text(
        json.dumps(settings, indent=2), encoding='utf-8')
    mask = None
    if keepout_enabled:
        mask = _write_map(state_dir / 'mask.yaml', _obstacle_pixels())
    return selected, mask
