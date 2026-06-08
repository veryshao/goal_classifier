import glob, json, os, shutil

from prepare_data import load_labels_from_json

BEST_MODEL_DIR = "results/best_model"


def has_BE_support(fold_name: str) -> bool:
    """
    True if the held-out conversation for this fold has at least one B and
    one E in its ground-truth labels — i.e. eval_f1_macro for that fold
    actually reflects boundary-detection performance rather than being
    inflated/deflated by a held-out session with no goal-discussion span.
    """
    stem = os.path.splitext(fold_name)[0]
    label_path = os.path.join("data/labels", f"{stem}.json")
    if not os.path.exists(label_path):
        return False
    values = load_labels_from_json(label_path).values()
    return "B" in values and "E" in values


rows = []
for fold_dir in sorted(glob.glob("results/*/")):
    fold_dir = fold_dir.rstrip("/")
    if not os.path.isdir(fold_dir) or os.path.basename(fold_dir) == "best_model":
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
    flag = "" if has_BE_support(name) else "  (no B/E in held-out labels — excluded)"
    print(f"{f1:.3f}  {name}  ({os.path.basename(ckpt)}){flag}")
print(f"\nmean macro-F1 across {len(rows)} folds: "
      f"{sum(f1 for _, f1, _ in rows) / len(rows):.3f}")

eligible = [row for row in rows if has_BE_support(row[0])]
if not eligible:
    raise SystemExit("No fold's held-out conversation has nonzero B/E support — "
                     "can't pick a best model on that basis.")

best_name, best_f1, best_ckpt = max(eligible, key=lambda r: r[1])
print(f"\nBest fold with nonzero B/E support: {best_name}  macro-F1={best_f1:.3f}")
print(f"Best checkpoint: {best_ckpt}")

# Overwrite any existing best_model so consecutive runs always reflect the
# current-best pick rather than accumulating/merging stale checkpoint files.
if os.path.lexists(BEST_MODEL_DIR):
    shutil.rmtree(BEST_MODEL_DIR)
shutil.copytree(best_ckpt, BEST_MODEL_DIR)
print(f"Copied {best_ckpt} -> {BEST_MODEL_DIR}")
