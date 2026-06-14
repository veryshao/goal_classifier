import glob, json, os, shutil

from prepare_data import load_labels_from_json

BEST_MODEL_DIR = "results/best_model"


def fold_stem(fold_name: str) -> str:
    """Strip ALL extensions from a fold directory basename.

    Fold dirs may be named '<stem>.md.txt' (created when transcripts still
    used that extension) or '<stem>.txt' — os.path.splitext only strips the
    last extension, so we loop until nothing remains.
    """
    name = fold_name
    while True:
        root, ext = os.path.splitext(name)
        if not ext:
            break
        name = root
    return name


def is_loo_fold(fold_name: str) -> bool:
    """True only if this results/ subdirectory corresponds to an actual LOO fold.

    Non-LOO directories (e.g. results/single/ from train_single.py) don't
    map to any transcript file, so they are excluded from fold statistics.
    """
    stem = fold_stem(fold_name)
    return os.path.exists(os.path.join("data/transcripts", f"{stem}.txt"))


def has_I_support(fold_name: str) -> bool:
    """True if the held-out conversation has at least one I label."""
    stem = fold_stem(fold_name)
    label_path = os.path.join("data/labels", f"{stem}.json")
    if not os.path.exists(label_path):
        return False
    return "I" in load_labels_from_json(label_path).values()


rows = []
for fold_dir in sorted(glob.glob("results/*/")):
    fold_dir = fold_dir.rstrip("/")
    fold_name = os.path.basename(fold_dir)
    if not os.path.isdir(fold_dir) or fold_name == "best_model":
        continue
    if not is_loo_fold(fold_name):
        continue
    ckpts = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if not ckpts:
        continue
    # The highest-step checkpoint holds the complete log_history / best_metric
    # / best_model_checkpoint for that fold's whole training run.
    latest = max(ckpts, key=lambda p: int(p.split("-")[-1]))
    state = json.load(open(os.path.join(latest, "trainer_state.json")))
    rows.append((
        os.path.basename(fold_dir),
        state["best_metric"],
        state["best_model_checkpoint"],
    ))

if not rows:
    raise SystemExit("No fold checkpoints found under results/ — run train.py first.")

for name, f1, ckpt in rows:
    flag = "" if has_I_support(name) else "  (no I labels in held-out fold — excluded)"
    print(f"{f1:.3f}  {name}  ({os.path.basename(ckpt)}){flag}")
print(f"\nmean macro-F1 across {len(rows)} folds: "
      f"{sum(f1 for _, f1, _ in rows) / len(rows):.3f}")

eligible = [row for row in rows if has_I_support(row[0])]
if not eligible:
    raise SystemExit("No fold's held-out conversation has any I labels — "
                     "can't pick a best model on that basis.")

best_name, best_f1, best_ckpt = max(eligible, key=lambda r: r[1])
print(f"\nBest fold with I-label support: {best_name}  macro-F1={best_f1:.3f}")
print(f"Best checkpoint: {best_ckpt}")

# Overwrite any existing best_model so consecutive runs always reflect the
# current-best pick rather than accumulating/merging stale checkpoint files.
if os.path.lexists(BEST_MODEL_DIR):
    shutil.rmtree(BEST_MODEL_DIR)
shutil.copytree(best_ckpt, BEST_MODEL_DIR)
print(f"Copied {best_ckpt} -> {BEST_MODEL_DIR}")
