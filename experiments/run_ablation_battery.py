"""
Run the full Phase-1 ablation grid from paper_a_star_submission_draft.md §8 (P0/P1) with multi-seed.

Each experiment writes under:
    {battery_root}/{slug}/seed_*/eval/phase1_results.json
    {battery_root}/{slug}/aggregated_results.json   (written by run_phase1_multi_seed)

Optional `--no_aggregate_per_run`: multi_seed leaves aggregation to this script (same files as above).

After each slug finishes, aggregates are kept while **seed_* trees** (splits, checkpoints,
per-seed eval) are removed by default to save disk before the next slug. Disable with
`--no_prune_seed_dirs` or use `--archive_slug_then_delete` to move JSON/MD to
`_summaries/{slug}/` and delete the whole slug folder.

Then compare all slugs with:
    python -m experiments.compare_ablation_aggregates --battery_root <same path>

Example (from repo root)
------------------------
python -m experiments.run_ablation_battery \\
  --candidate_cache cache/mpdoc_val_full \\
  --battery_root multi_seed_runs/paper_ablation_battery \\
  --num_epochs 20 --train_batch_size 16 --lr 5e-5 --top_k 20 --eval_k 10

Dry-run:
    ... --dry_run
Run only a subset:
    ... --only region_main_default,page_gat_default
Resume (skip slugs that already have aggregated_results.json):
    ... --skip_completed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# slug -> CLI overrides (everything else comes from shared args)
ABLATION_SPECS: List[Tuple[str, Dict[str, Any]]] = [
    # P0: page family (eval emits GAT + no-graph + baselines)
    (
        "page_gat_default",
        {"reranker_type": "page", "graph_preset": "default"},
    ),
    (
        "page_gat_sparse_graph",
        {"reranker_type": "page", "graph_preset": "sparse-graph"},
    ),
    # Main region setting (match paper defaults)
    (
        "region_main_default",
        {
            "reranker_type": "region",
            "graph_preset": "default",
            "grid_rows": 2,
            "grid_cols": 2,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.15,
            "lambda_mix_end": 0.55,
            "lambda_mix_warmup_steps": 1200,
        },
    ),
    (
        "region_sparse_graph",
        {
            "reranker_type": "region",
            "graph_preset": "sparse-graph",
            "grid_rows": 2,
            "grid_cols": 2,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.15,
            "lambda_mix_end": 0.55,
            "lambda_mix_warmup_steps": 1200,
        },
    ),
    # P1: region grid
    (
        "region_grid_1x1",
        {
            "reranker_type": "region",
            "graph_preset": "default",
            "grid_rows": 1,
            "grid_cols": 1,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.15,
            "lambda_mix_end": 0.55,
            "lambda_mix_warmup_steps": 1200,
        },
    ),
    (
        "region_grid_3x3",
        {
            "reranker_type": "region",
            "graph_preset": "default",
            "grid_rows": 3,
            "grid_cols": 3,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.15,
            "lambda_mix_end": 0.55,
            "lambda_mix_warmup_steps": 1200,
        },
    ),
    # P1: λ schedule (flat = fixed interpolation weight during anneal target)
    (
        "region_lambda_fixed_mid",
        {
            "reranker_type": "region",
            "graph_preset": "default",
            "grid_rows": 2,
            "grid_cols": 2,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.35,
            "lambda_mix_end": 0.35,
            "lambda_mix_warmup_steps": 1200,
        },
    ),
    # P0: second split regime (more train)
    (
        "region_split_70_15_15",
        {
            "reranker_type": "region",
            "graph_preset": "default",
            "grid_rows": 2,
            "grid_cols": 2,
            "sem_threshold_region": 0.70,
            "lambda_mix_start": 0.15,
            "lambda_mix_end": 0.55,
            "lambda_mix_warmup_steps": 1200,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    ),
]


def _build_multi_seed_cmd(
    exe: str,
    candidate_cache: str,
    root: Path,
    seeds: str,
    num_epochs: int,
    train_batch_size: int,
    lr: float,
    top_k: int,
    eval_k: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    no_aggregate: bool,
    dry_run: bool,
    overrides: Dict[str, Any],
) -> List[str]:
    cmd: List[str] = [
        exe,
        "-m",
        "experiments.run_phase1_multi_seed",
        "--candidate_cache",
        candidate_cache,
        "--root",
        str(root),
        "--seeds",
        seeds,
        "--num_epochs",
        str(num_epochs),
        "--train_batch_size",
        str(train_batch_size),
        "--lr",
        str(lr),
        "--top_k",
        str(top_k),
        "--eval_k",
        str(eval_k),
        "--train_ratio",
        str(train_ratio),
        "--val_ratio",
        str(val_ratio),
        "--test_ratio",
        str(test_ratio),
        "--eval_dir_name",
        "eval",
    ]
    if no_aggregate:
        cmd.append("--no_aggregate")

    # Apply overrides (reranker_type, graph_preset, grid_*, lambdas, etc.)
    # Must use underscores to match argparse on run_phase1_multi_seed (e.g. --reranker_type).
    for key, val in overrides.items():
        if val is None:
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.append(flag)
            cmd.append(str(val))

    return cmd


def _run(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _prune_seed_directories(run_root: Path, dry_run: bool) -> int:
    """Remove every seed_* directory under run_root. Returns count removed."""
    n = 0
    for p in sorted(run_root.glob("seed_*")):
        if p.is_dir():
            print(f"  [prune] removing {p}", flush=True)
            if not dry_run:
                shutil.rmtree(p)
            n += 1
    return n


def _archive_summaries_and_remove_run_root(
    battery: Path,
    run_root: Path,
    slug: str,
    dry_run: bool,
) -> Path:
    """
    Copy aggregated_results.json and aggregated_report.md to battery/_summaries/{slug}/,
    then delete run_root entirely.
    """
    dest = battery / "_summaries" / slug
    if dry_run:
        print(f"  [dry_run] would copy aggregates to {dest} and remove {run_root}", flush=True)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("aggregated_results.json", "aggregated_report.md"):
        src = run_root / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    shutil.rmtree(run_root)
    print(f"  [prune] archived summaries to {dest} and removed {run_root}", flush=True)
    return dest


def _aggregate_one(
    exe: str,
    run_root: Path,
    eval_dir_name: str,
    dry_run: bool,
) -> None:
    pattern = str(run_root / "seed_*" / eval_dir_name / "phase1_results.json")
    out_json = run_root / "aggregated_results.json"
    out_md = run_root / "aggregated_report.md"
    cmd = [
        exe,
        "-m",
        "experiments.summarize_phase1_multi_seed",
        "--glob",
        pattern,
        "--out_json",
        str(out_json),
        "--out_md",
        str(out_md),
    ]
    _run(cmd, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run multi-seed ablation battery; one subdirectory per condition.",
    )
    ap.add_argument("--candidate_cache", required=True)
    ap.add_argument(
        "--battery_root",
        default="multi_seed_runs/paper_ablation_battery",
        help="Parent directory; each experiment uses {battery_root}/{slug}/",
    )
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--train_batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--train_ratio", type=float, default=0.12)
    ap.add_argument("--val_ratio", type=float, default=0.08)
    ap.add_argument("--test_ratio", type=float, default=0.80)
    ap.add_argument(
        "--no_aggregate_per_run",
        action="store_true",
        help="Pass --no_aggregate to multi_seed; battery will call summarize after each run (useful to force ordering).",
    )
    ap.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip a slug if aggregated_results.json already exists in its run root.",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated slugs to run (default: all known ablations).",
    )
    ap.add_argument(
        "--no_prune_seed_dirs",
        action="store_true",
        help="Keep seed_*/ directories (large) after each slug; default is to delete them after aggregates exist.",
    )
    ap.add_argument(
        "--archive_slug_then_delete",
        action="store_true",
        help="After aggregating, copy JSON/Markdown to {battery_root}/_summaries/{slug}/ and remove the whole slug directory (max savings; compare_ablation_aggregates reads _summaries too).",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    only: Optional[set] = None
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}

    exe = sys.executable
    battery = Path(args.battery_root).expanduser().resolve()
    battery.mkdir(parents=True, exist_ok=True)

    manifest_path = battery / "battery_manifest.json"
    manifest_rows: List[Dict[str, Any]] = []

    for slug, overrides in ABLATION_SPECS:
        if only is not None and slug not in only:
            continue
        run_root = battery / slug

        agg_path = run_root / "aggregated_results.json"
        arch_agg = battery / "_summaries" / slug / "aggregated_results.json"
        if args.skip_completed and (agg_path.is_file() or arch_agg.is_file()):
            found = agg_path if agg_path.is_file() else arch_agg
            print(f"[skip] {slug} (found {found})", flush=True)
            manifest_rows.append({
                "slug": slug,
                "root": str(run_root if agg_path.is_file() else arch_agg.parent),
                "status": "skipped_completed",
            })
            continue

        tr, vr, te = args.train_ratio, args.val_ratio, args.test_ratio
        # Per-experiment split overrides
        o = dict(overrides)
        if "train_ratio" in o:
            tr = float(o.pop("train_ratio"))
        if "val_ratio" in o:
            vr = float(o.pop("val_ratio"))
        if "test_ratio" in o:
            te = float(o.pop("test_ratio"))

        cmd = _build_multi_seed_cmd(
            exe=exe,
            candidate_cache=str(Path(args.candidate_cache).expanduser().resolve()),
            root=run_root,
            seeds=args.seeds,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
            top_k=args.top_k,
            eval_k=args.eval_k,
            train_ratio=tr,
            val_ratio=vr,
            test_ratio=te,
            no_aggregate=args.no_aggregate_per_run,
            dry_run=args.dry_run,
            overrides=o,
        )
        print(f"\n=== Battery run: {slug} ===", flush=True)
        _run(cmd, args.dry_run)

        if args.no_aggregate_per_run and not args.dry_run:
            _aggregate_one(exe, run_root, "eval", dry_run=False)

        # Drop heavy per-seed trees once summaries exist (before the next slug starts).
        if not args.dry_run:
            if not agg_path.is_file():
                print(
                    f"  [warn] skip disk cleanup for {slug}: missing {agg_path}",
                    flush=True,
                )
            elif args.archive_slug_then_delete:
                dest = _archive_summaries_and_remove_run_root(
                    battery, run_root, slug, dry_run=False
                )
                manifest_rows.append({
                    "slug": slug,
                    "root": str(dest),
                    "status": "ran_archived",
                    "overrides": overrides,
                })
                continue
            elif not args.no_prune_seed_dirs:
                n_rm = _prune_seed_directories(run_root, dry_run=False)
                if n_rm:
                    print(f"  [prune] removed {n_rm} seed_* tree(s) under {run_root}", flush=True)

        manifest_rows.append({"slug": slug, "root": str(run_root), "status": "ran", "overrides": overrides})

    if args.dry_run:
        print("\n(dry_run: no manifest written)")
        return

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_cache": str(Path(args.candidate_cache).expanduser().resolve()),
                "battery_root": str(battery),
                "seeds": args.seeds,
                "runs": manifest_rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote manifest {manifest_path}", flush=True)
    print(
        "Compare all runs:\n  python -m experiments.compare_ablation_aggregates "
        f"--battery_root {battery}",
        flush=True,
    )


if __name__ == "__main__":
    main()
