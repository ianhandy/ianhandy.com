#!/usr/bin/env python3
"""Preset generator and exporter for Celestial Sim.

Usage:
    python3 generate_presets.py export [--out presets.json]
        Dump the canonical preset table (matches PRESETS in sim.js) as JSON.

    python3 generate_presets.py random [--count N] [--seed S]
        Build a random "stable cluster" preset: a heavy central body with
        N satellites on circular orbits. Speed is set so each orbit closes:
            v = sqrt(M / r)     (G = 1 in sim units)
        Prints both a JSON dump and a JS literal ready to paste into sim.js.

The CLI banner reads strings.json so this tool stays in sync with the web UI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRINGS_PATH = HERE / "strings.json"

# Mirror of PRESETS in sim.js. Keep in sync by hand, or regenerate via export.
CANONICAL_PRESETS = {
    "binary": {
        "bodies": [
            {"pos": [-1.0, 0.0], "vel": [0.0, -0.5], "mass": 1.0},
            {"pos": [ 1.0, 0.0], "vel": [0.0,  0.5], "mass": 1.0},
        ],
        "colors": ["#f6c667", "#ff7ab6"],
    },
    "figure8": {
        "bodies": [
            {"pos": [-0.97000436,  0.24308753], "vel": [ 0.466203685,  0.43236573], "mass": 1.0},
            {"pos": [ 0.97000436, -0.24308753], "vel": [ 0.466203685,  0.43236573], "mass": 1.0},
            {"pos": [ 0.0,         0.0       ], "vel": [-0.93240737,  -0.86473146], "mass": 1.0},
        ],
        "colors": ["#89c9ff", "#f6c667", "#ff7ab6"],
    },
    "solar": {
        "bodies": [
            {"pos": [ 0.0, 0.0], "vel": [ 0.0,    0.0  ], "mass": 20.0},
            {"pos": [ 1.5, 0.0], "vel": [ 0.0,    3.651], "mass":  0.3},
            {"pos": [-2.5, 0.0], "vel": [ 0.0,   -2.828], "mass":  0.6},
            {"pos": [ 0.0, 4.0], "vel": [-2.236,  0.0  ], "mass":  0.2},
        ],
        "colors": ["#f6c667", "#6be1c7", "#ff7ab6", "#c89bf5"],
    },
    "lagrange": {
        "bodies": [
            {"pos": [ 1.000,  0.000], "vel": [ 0.000,  0.760], "mass": 1.0},
            {"pos": [-0.500,  0.866], "vel": [-0.658, -0.380], "mass": 1.0},
            {"pos": [-0.500, -0.866], "vel": [ 0.658, -0.380], "mass": 1.0},
        ],
        "colors": ["#f6c667", "#89c9ff", "#ff7ab6"],
    },
    "pinwheel": {
        "bodies": [
            {"pos": [ 0.0,  1.0], "vel": [-0.978,  0.000], "mass": 1.0},
            {"pos": [ 1.0,  0.0], "vel": [ 0.000,  0.978], "mass": 1.0},
            {"pos": [ 0.0, -1.0], "vel": [ 0.978,  0.000], "mass": 1.0},
            {"pos": [-1.0,  0.0], "vel": [ 0.000, -0.978], "mass": 1.0},
        ],
        "colors": ["#f6c667", "#89c9ff", "#ff7ab6", "#80e0a3"],
    },
}

PALETTE = ["#f6c667", "#89c9ff", "#ff7ab6", "#6be1c7", "#c89bf5", "#80e0a3"]


def load_strings() -> dict:
    try:
        return json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def print_banner(strings: dict) -> None:
    app = strings.get("app", {})
    print(app.get("brand", "Celestial Sim"))
    tagline = app.get("tagline")
    if tagline:
        print(tagline)


def circular_orbit_preset(satellite_count: int, rng: random.Random) -> dict:
    """Heavy central body with `satellite_count` satellites on circular orbits."""
    central_mass = round(rng.uniform(15.0, 25.0), 3)
    bodies = [{"pos": [0.0, 0.0], "vel": [0.0, 0.0], "mass": central_mass}]

    for i in range(satellite_count):
        radius = 1.2 + i * 0.9 + rng.uniform(-0.1, 0.1)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        speed = math.sqrt(central_mass / radius)
        # Velocity is perpendicular to the radius vector (counter-clockwise).
        bodies.append({
            "pos": [round(radius * math.cos(phase), 4),
                    round(radius * math.sin(phase), 4)],
            "vel": [round(-speed * math.sin(phase), 4),
                    round( speed * math.cos(phase), 4)],
            "mass": round(rng.uniform(0.15, 0.9), 3),
        })

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(bodies))]
    return {"bodies": bodies, "colors": colors}


def format_js_literal(preset: dict) -> str:
    """Render a preset as a JS object literal for pasting into sim.js PRESETS."""
    lines = ["generated: {", "  bodies: ["]
    for body in preset["bodies"]:
        px, py = body["pos"]
        vx, vy = body["vel"]
        lines.append(
            f"    {{ pos: [{px:>8}, {py:>8}], "
            f"vel: [{vx:>8}, {vy:>8}], "
            f"mass: {body['mass']} }},"
        )
    lines.append("  ],")
    color_list = ", ".join(f"'{c}'" for c in preset["colors"])
    lines.append(f"  colors: [{color_list}],")
    lines.append("},")
    return "\n".join(lines)


def cmd_export(args: argparse.Namespace) -> int:
    args.out.write_text(json.dumps(CANONICAL_PRESETS, indent=2), encoding="utf-8")
    print(f"wrote {len(CANONICAL_PRESETS)} presets -> {args.out}")
    return 0


def cmd_random(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    preset = circular_orbit_preset(args.count, rng)
    print("# JSON")
    print(json.dumps(preset, indent=2))
    print()
    print("# JS literal (paste into sim.js PRESETS)")
    print(format_js_literal(preset))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    export_cmd = sub.add_parser("export", help="dump canonical presets as JSON")
    export_cmd.add_argument("--out", type=Path, default=HERE / "presets.json")
    export_cmd.set_defaults(func=cmd_export)

    random_cmd = sub.add_parser("random", help="generate a random stable-orbit preset")
    random_cmd.add_argument("--count", type=int, default=4,
                            help="number of satellites around the central body")
    random_cmd.add_argument("--seed", type=int, default=None)
    random_cmd.set_defaults(func=cmd_random)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    print_banner(load_strings())
    print()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
