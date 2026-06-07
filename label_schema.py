# Label schema for goal conversation boundary detection
# Applied at the UTTERANCE level only (non-utterance events provide context)

LABEL2ID = {
    "O": 0,   # Outside any goal conversation
    "B": 1,   # First utterance OF a goal conversation
    "I": 2,   # Inside a goal conversation (not start/end)
    "E": 3,   # Last utterance OF a goal conversation
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Labeling rules (for human annotators):
# B: The utterance where tutor first mentions goals/progress AND/OR
#    immediately precedes a PLUS Students screen share
#    e.g. "before we start...I'll share my screen. We can look at some of the goals"
#
# E: The utterance after which the tutor says "log into your math software"
#    or the screen switches away from PLUS Students back to math content
#
# I: Any utterance between B and E (slider discussions, skill counts, etc.)
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