"""Single source of truth for the active annotation scheme.

LABEL2ID, DEFAULT_LABEL, POSITIVE_LABEL, ID2LABEL, LABEL_NAMES, and LABEL_IDS
define the active scheme (currently U/R) consumed by train.py, evaluate.py,
predict.py, embed.py, bert_embed.py, and their prediction scripts.

ANNOTATION_SCHEMES documents both annotation sets (O/I and U/R) for tools that
process multiple schemes at once (goal_stats.py).

To switch the active scheme, edit only LABEL2ID, DEFAULT_LABEL, and POSITIVE_LABEL
here — all other files derive from these constants. See CLAUDE.md for steps.
"""
# Label schema for goal conversation detection
# Applied at the UTTERANCE level only (non-utterance events provide context)
#
# Two parallel annotation sets exist in this repo, each with its own label
# directories (see ANNOTATION_SCHEMES below):
#   O/I — Outside / Inside a goal-setting discussion
#         (data/labels/ and data/eval_labels/)
#   U/R — Unrelated / Related to goal discussion, R marking goal utterances
#         (data/binary_labels/ and data/eval_binary_labels/)
# LABEL2ID and the constants derived from it define the ACTIVE scheme used by
# the training/eval/predict pipelines; to switch, follow the steps in
# CLAUDE.md. Analysis tools that need both sets at once (goal_stats.py)
# read ANNOTATION_SCHEMES instead of the active-scheme constants.

LABEL2ID = {
    "U": 0,   # Outside any goal-setting discussion
    "R": 1,   # Inside a goal-setting discussion
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEFAULT_LABEL  = "U"
POSITIVE_LABEL = "R"
LABEL_NAMES    = [ID2LABEL[i] for i in range(len(ID2LABEL))]
LABEL_IDS      = list(range(len(LABEL2ID)))

# Both annotation sets, independent of which one is active above.
ANNOTATION_SCHEMES = {
    "OI": {
        "description":    "Outside / Inside a goal-setting discussion",
        "positive_label": "I",
        "default_label":  "O",
        "label_dirs":     ["data/labels", "data/eval_labels"],
    },
    "UR": {
        "description":    "Unrelated / Related to goal discussion",
        "positive_label": "R",
        "default_label":  "U",
        "label_dirs":     ["data/binary_labels", "data/eval_binary_labels"],
    },
}

# Labeling rules (for human annotators):
# I: Any utterance that is part of a goal-setting discussion — includes the
#    opening mention of goals, slider/skill-count discussion, and the closing
#    transition back to math content.
#    e.g. "before we start I'll share my screen so we can look at your goals"
#    e.g. "the system recommends 30 minutes this week"
#    e.g. "okay let's get into the math now"  (if still within the span)
#
# O: Everything else (math problem solving, greetings unrelated to goals, etc.)

# Key app signals that strongly correlate with goal conversations:
GOAL_APP_SIGNALS = [
    "PLUS Students",              # screen share of goal dashboard
    "Update Goals button",        # mouse click target
    "Save Goals button",          # mouse click target
    "slider for 'Minutes",        # drag on goals slider
    "slider for 'Skills",         # drag on goals slider
    "PLUS Tutoring",              # alternative dashboard name seen in transcripts
]

GOAL_VERBAL_SIGNALS = [
    "goal", "goals", "progress", "minutes", "skill", "skills",
    "system recommends", "reduce", "achieve", "share my screen",
    "track", "completed", "percent"
]