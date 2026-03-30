"""
Run Phase 1 disjoint splits + train + eval for multiple RNG seeds, then optionally aggregate.

Per seed *s* (default five seeds: 42–46):
  {root}/seed_{s}/splits/{train,val,test}  — from make_disjoint_splits
  {root}/seed_{s}/checkpoints/             — train output (best.pt)
  {root}/seed_{s}/eval/phase1_results.json — eval output

After all seeds, run the summarizer (or use --no_aggregate and call summarize yourself).

Example
-------
python -m experiments.run_phase1_multi_seed \\
    --candidate_cache cache/mpdoc_val_full \\
    --root multi_seed_runs/mpdoc_val_disjoint \\
    --graph_preset sparse-graph \\
    --num_epochs 10

python -m experiments.summarize_phase1_multi_seed \\
    --glob 'multi_seed_runs/mpdoc_val_disjoint/seed_*/eval/phase1_results.json' \\
    --out_json multi_seed_runs/mpdoc_val_disjoint/aggregated_results.json \\
    --out_md multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(int(part))
    if len(out) < 1:
        raise ValueError("Provide at least one seed.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed disjoint split + train + eval for Phase 1.")
    ap.add_argument(
        "--candidate_cache",
        required=True,
        help="Directory with meta.json + *.pt (e.g. cache/mpdoc_val_full from repo root). Not a placeholder path.",
    )
    ap.add_argument("--root", default="multi_seed_runs/run", help="Work directory; seed_* subdirs created")
    ap.add_argument("--seeds", default="42,43,44,45,46", help="Comma/space-separated seeds")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--skip_splits", action="store_true", help="Assume splits already exist under each seed_*")
    ap.add_argument("--skip_train", action="store_true", help="Skip training (reuse checkpoints)")
    ap.add_argument("--skip_eval", action="store_true", help="Skip eval only")
    ap.add_argument("--graph_preset", default="sparse-graph", choices=["default", "sparse-graph"])
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=20, help="Must match cache top_k used when building candidate_cache")
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true", help="Only print commands")
    ap.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Do not run summarize_phase1_multi_seed after all seeds",
    )
    args = ap.parse_args()

    cand = Path(args.candidate_cache).expanduser().resolve()
    if not args.skip_splits:
        meta = cand / "meta.json"
        if not meta.is_file():
            print(
                "ERROR: --candidate_cache must be a real cache folder containing meta.json.\n"
                f"  Missing: {meta}\n"
                "  Example (from repo root): --candidate_cache cache/mpdoc_val_full\n"
                "  Do not use placeholder paths from docs (e.g. /đường/dẫn/...).",
                file=sys.stderr,
            )
            raise SystemExit(1)

    exe = sys.executable
    root = Path(args.root).resolve()
    seeds = parse_seeds(args.seeds)

    for seed in seeds:
        sdir = root / f"seed_{seed}"
        split_root = sdir / "splits"
        train_c = split_root / "train"
        val_c = split_root / "val"
        test_c = split_root / "test"
        ckpt_dir = sdir / "checkpoints"
        eval_dir = sdir / "eval"

        if not args.skip_splits:
            split_root.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    exe,
                    "-m",
                    "experiments.train.make_disjoint_splits",
                    "--candidate_cache",
                    str(cand),
                    "--output_root",
                    str(split_root.resolve()),
                    "--train_ratio",
                    str(args.train_ratio),
                    "--val_ratio",
                    str(args.val_ratio),
                    "--test_ratio",
                    str(args.test_ratio),
                    "--seed",
                    str(seed),
                ],
                args.dry_run,
            )

        if not args.skip_train:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    exe,
                    "-m",
                    "experiments.run_phase1",
                    "train",
                    "--train_cache",
                    str(train_c.resolve()),
                    "--val_cache",
                    str(val_c.resolve()),
                    "--output_dir",
                    str(ckpt_dir.resolve()),
                    "--num_epochs",
                    str(args.num_epochs),
                    "--batch_size",
                    str(args.train_batch_size),
                    "--top_k",
                    str(args.top_k),
                    "--graph_preset",
                    args.graph_preset,
                ],
                args.dry_run,
            )

        if not args.skip_eval:
            eval_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best.pt"
            if not args.dry_run and not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path} (train or copy weights first)")
            _run(
                [
                    exe,
                    "-m",
                    "experiments.run_phase1",
                    "eval",
                    "--eval_cache",
                    str(test_c.resolve()),
                    "--checkpoint",
                    str(ckpt_path.resolve()),
                    "--output_dir",
                    str(eval_dir.resolve()),
                    "--k",
                    str(args.eval_k),
                    "--graph_preset",
                    args.graph_preset,
                ],
                args.dry_run,
            )

    if args.no_aggregate or args.dry_run:
        return

    pattern = str(root / "seed_*" / "eval" / "phase1_results.json")
    agg_json = root / "aggregated_results.json"
    agg_md = root / "aggregated_report.md"
    _run(
        [
            exe,
            "-m",
            "experiments.summarize_phase1_multi_seed",
            "--glob",
            pattern,
            "--out_json",
            str(agg_json),
            "--out_md",
            str(agg_md),
        ],
        False,
    )


if __name__ == "__main__":
    main()
