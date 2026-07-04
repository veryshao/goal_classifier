# Label schema for goal conversation detection
# Applied at the UTTERANCE level only (non-utterance events provide context)

LABEL2ID = {
    "O": 0,   # Outside any goal-setting discussion
    "I": 1,   # Inside a goal-setting discussion
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEFAULT_LABEL  = "O"
POSITIVE_LABEL = "I"
LABEL_NAMES    = [ID2LABEL[i] for i in range(len(ID2LABEL))]
LABEL_IDS      = list(range(len(LABEL2ID)))

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