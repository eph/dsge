from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def _resolve_octave_dir(linver_root: Path) -> Path:
    if linver_root.name == "octave":
        return linver_root
    cand = linver_root / "octave"
    if cand.exists():
        return cand
    raise SystemExit(
        f"Could not find LINVER `octave/` directory under {linver_root}. "
        "Pass the extracted LINVER root (containing `octave/`) or the `octave/` directory itself."
    )


def _write_dynare_stub(stub_dir: Path) -> None:
    # `make_runmod_octave.m` unconditionally calls `dynare runmod ...` at the end.
    # For translation checks we only need the generated `runmod.mod`, so we provide a stub.
    (stub_dir / "dynare.m").write_text(
        "function varargout = dynare(varargin)\n"
        "  disp('dynare (stub) called; skipping Dynare parsing');\n"
        "  varargout = cell(1, nargout);\n"
        "end\n",
        encoding="utf-8",
    )


def _run_octave_make_runmod(
    *,
    octave_dir: Path,
    expvers: str,
    mprule: str,
    elb_imposed: str,
    stub_dir: Path | None,
    log_path: Path,
) -> None:
    eval_parts = [f"cd('{octave_dir.as_posix()}');"]
    if stub_dir is not None:
        eval_parts.append(f"addpath('{stub_dir.as_posix()}');")
    eval_parts.extend(
        [
            f"expvers='{expvers}';",
            f"mprule='{mprule}';",
            f"elb_imposed='{elb_imposed}';",
            "make_parameters_octave;",
            "if strcmp(fail_flag,'yes'); error('make_parameters_octave failed'); end;",
            "make_runmod_octave;",
            "if strcmp(fail_flag,'yes'); error('make_runmod_octave failed'); end;",
        ]
    )

    cmd = ["octave", "--quiet", "--norc", "--eval", " ".join(eval_parts)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"Octave failed (see {log_path}).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate LINVER `runmod.mod` Dynare files using the Fed-provided Octave scripts.\n\n"
            "This runs `make_parameters_octave.m` + `make_runmod_octave.m` for one or more `expvers` values.\n"
            "By default we install a temporary `dynare.m` stub so the script can run even if Dynare is not on the Octave path."
        )
    )
    ap.add_argument(
        "linver_root",
        type=Path,
        help="Path to the extracted Fed LINVER package root (contains `octave/`) or the `octave/` directory itself.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/linver_runmod_mods"),
        help="Output directory for generated `.mod` files (default: /tmp/linver_runmod_mods).",
    )
    ap.add_argument(
        "--expvers",
        nargs="+",
        default=["mcap", "mcapwp", "var"],
        help="Expectational versions to generate (default: mcap mcapwp var).",
    )
    ap.add_argument("--mprule", default="intay", help="Monetary policy rule (default: intay).")
    ap.add_argument("--elb-imposed", default="no", choices=["yes", "no"], help="Impose ELB constraint (default: no).")
    ap.add_argument(
        "--use-dynare",
        action="store_true",
        help="Do not install a dynare stub; requires Dynare to be on the Octave path.",
    )
    args = ap.parse_args()

    octave_dir = _resolve_octave_dir(args.linver_root)
    for req in ["make_parameters_octave.m", "make_runmod_octave.m"]:
        if not (octave_dir / req).exists():
            raise SystemExit(f"Missing required file {octave_dir / req}.")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="linver_dynare_stub_") as td:
        stub_dir = Path(td)
        if args.use_dynare:
            stub = None
        else:
            _write_dynare_stub(stub_dir)
            stub = stub_dir

        for expvers in args.expvers:
            log_path = out_dir / f"runmod_{expvers}_{args.mprule}.log"
            _run_octave_make_runmod(
                octave_dir=octave_dir,
                expvers=expvers,
                mprule=args.mprule,
                elb_imposed=args.elb_imposed,
                stub_dir=stub,
                log_path=log_path,
            )

            src = octave_dir / "runmod.mod"
            if not src.exists():
                raise SystemExit(f"Expected {src} to exist after Octave run (see {log_path}).")

            dst = out_dir / f"runmod_{expvers}_{args.mprule}.mod"
            shutil.copyfile(src, dst)
            print(f"Wrote {dst}")


if __name__ == "__main__":
    main()

