"""
Grid search over window size and timestamp window for the embedding pipelines.

Runs embed.py or bert_embed.py for every combination of --window and
--timestamp-window, writing each run's outputs to a separate subdirectory,
then collects macro-F1 and positive-class F1 into a summary CSV.

Usage:
  python sweep.py                                  # embed pipeline, default grid
  python sweep.py --pipeline bert_embed            # BERT embedding pipeline
  python sweep.py --windows 2 5 10 --timestamp-windows 0 15 30 60
  python sweep.py --pipeline both                  # run both pipelines

  0 in --timestamp-windows means index-based (no --timestamp-window flag passed).

Output:
  sweep_results/<pipeline>/summary.csv
  sweep_results/<pipeline>/w<W>_ts<TS>/   (one subdir per run, with full eval outputs)

Note: bert_embed loads the BERT model in each subprocess, adding ~30s overhead
per run. For large grids consider running embed first to prototype the grid.
"""
import argparse
import csv
import os
import subprocess
import sys
from itertools import product

from label_schema import POSITIVE_LABEL

DEFAULT_WINDOWS           = [2, 5, 10]
DEFAULT_TIMESTAMP_WINDOWS = [0, 15, 30, 60]  # 0 = index-based


# ── Result parsing ────────────────────────────────────────────────────────────

def parse_results(output_dir: str) -> dict:
    """
    Read classification_report.txt from output_dir and return a dict with
    positive_f1 and macro_f1. Returns None values if the file is missing.
    """
    path = os.path.join(output_dir, "classification_report.txt")
    if not os.path.exists(path):
        return {"positive_f1": None, "macro_f1": None}

    with open(path) as f:
        lines = f.readlines()

    positive_f1 = None
    macro_f1    = None
    for line in lines:
        parts = line.split()
        if parts and parts[0] == POSITIVE_LABEL and len(parts) >= 4:
            try:
                positive_f1 = float(parts[3])
            except ValueError:
                pass
        if "macro avg" in line and len(parts) >= 5:
            try:
                macro_f1 = float(parts[4])
            except ValueError:
                pass

    return {"positive_f1": positive_f1, "macro_f1": macro_f1}


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(pipeline: str, window: int, ts: int,
            output_dir: str, train_labels: str, eval_labels: str) -> bool:
    script = f"{pipeline}.py"
    cmd = [
        sys.executable, script,
        "--window",      str(window),
        "--output-dir",  output_dir,
        "--train-labels", train_labels,
        "--eval-labels",  eval_labels,
    ]
    if ts > 0:
        cmd += ["--timestamp-window", str(ts)]

    label = f"window={window}  ts={'index' if ts == 0 else ts}s"
    print(f"\n{'─'*60}")
    print(f"  {pipeline}  |  {label}")
    print(f"  output → {output_dir}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] returncode={result.returncode}")
        return False
    return True


# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep(pipeline: str, windows: list[int], timestamp_windows: list[int],
          train_labels: str, eval_labels: str, base_dir: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    rows = []

    for window, ts in product(windows, timestamp_windows):
        ts_label  = "index" if ts == 0 else f"ts{ts}"
        run_dir   = os.path.join(base_dir, f"w{window}_{ts_label}")
        succeeded = run_one(pipeline, window, ts, run_dir, train_labels, eval_labels)
        metrics   = parse_results(run_dir) if succeeded else {"positive_f1": None, "macro_f1": None}
        rows.append({
            "pipeline":        pipeline,
            "window":          window,
            "timestamp_window": ts if ts > 0 else "index",
            "positive_f1":     metrics["positive_f1"],
            "macro_f1":        metrics["macro_f1"],
            "output_dir":      run_dir,
        })

    # ── Write summary CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(base_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pipeline", "window", "timestamp_window",
                           "positive_f1", "macro_f1", "output_dir"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  SWEEP SUMMARY  —  {pipeline}")
    print(f"{'═'*70}")
    print(f"  {'window':>6}  {'ts_window':>10}  {'pos_F1':>7}  {'macro_F1':>9}")
    print(f"  {'------':>6}  {'----------':>10}  {'-------':>7}  {'---------':>9}")
    for r in sorted(rows, key=lambda x: -(x["positive_f1"] or -1)):
        pos  = f"{r['positive_f1']:.3f}" if r["positive_f1"] is not None else "  —  "
        mac  = f"{r['macro_f1']:.3f}"    if r["macro_f1"]    is not None else "  —  "
        print(f"  {r['window']:>6}  {str(r['timestamp_window']):>10}  {pos:>7}  {mac:>9}")
    print(f"\n  Summary CSV → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search over window size and timestamp window."
    )
    parser.add_argument(
        "--pipeline", default="embed", choices=["embed", "bert_embed", "both"],
        help="Which pipeline to sweep (default: embed)."
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
        help=f"Window sizes to test (default: {DEFAULT_WINDOWS})."
    )
    parser.add_argument(
        "--timestamp-windows", type=int, nargs="+", default=DEFAULT_TIMESTAMP_WINDOWS,
        help=f"Timestamp windows in seconds to test; 0 = index-based "
             f"(default: {DEFAULT_TIMESTAMP_WINDOWS})."
    )
    parser.add_argument("--train-labels", default="data/labels",
                        help="Training label directory (default: data/labels).")
    parser.add_argument("--eval-labels", default="data/eval_labels",
                        help="Eval label directory (default: data/eval_labels).")
    parser.add_argument("--output-base", default="sweep_results",
                        help="Base directory for all sweep outputs (default: sweep_results).")
    args = parser.parse_args()

    pipelines = ["embed", "bert_embed"] if args.pipeline == "both" else [args.pipeline]

    total = len(pipelines) * len(args.windows) * len(args.timestamp_windows)
    print(f"Running {total} configuration(s) across pipeline(s): {pipelines}")
    print(f"  windows:           {args.windows}")
    print(f"  timestamp_windows: {args.timestamp_windows}")

    for pl in pipelines:
        base = os.path.join(args.output_base, pl)
        sweep(pl, args.windows, args.timestamp_windows,
              args.train_labels, args.eval_labels, base)
