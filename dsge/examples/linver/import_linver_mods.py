from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dsge.dynare_parser import parse_mod_file
from dsge.dynare_translate import to_yaml_like


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Dynare .mod files (LINVER) into dsge YAML format.")
    ap.add_argument("mod_dir", type=Path, help="Directory containing generated .mod files.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dsge/examples/linver/_imported"),
        help="Output directory for imported YAML files.",
    )
    args = ap.parse_args()

    mod_dir: Path = args.mod_dir
    out_dir: Path = args.out

    mods = sorted(mod_dir.glob("*.mod"))
    if not mods:
        raise SystemExit(f"No .mod files found in {mod_dir}.")

    out_dir.mkdir(parents=True, exist_ok=True)

    for mod_path in mods:
        parsed = parse_mod_file(mod_path)
        yaml_like = to_yaml_like(parsed, name=mod_path.stem)
        out_path = out_dir / f"{mod_path.stem}.yaml"
        out_path.write_text(yaml.safe_dump(yaml_like, sort_keys=False), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

