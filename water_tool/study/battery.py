"""View 2 context battery (design Section 6).

A small FIXED set of frame sentences applied to every probe word, so the
contextual next-token distribution is comparable across words. Three frame
families supply different kinds of situation:

  neutral    - bare topical mention, minimal situational pull
  perceptual - situates the word in lived/sensory experience
  relational - situates the word in a systemic/definitional relation

The View 1 -> View 2 divergence (how far contextual continuations depart
from the systemic neighbors) is the quantity of interest. Using identical
frames for every word keeps that divergence comparable.

Each frame has a `{w}` slot. The probe position for next-token prediction is
the end of the frame (we read the distribution over the token that follows).
"""

FRAMES: list[dict] = [
    {"id": "neutral_topic",   "family": "neutral",    "template": "The {w} was"},
    {"id": "neutral_def",     "family": "neutral",    "template": "A {w} is a"},
    {"id": "percept_sense",   "family": "perceptual", "template": "She could feel the {w}"},
    {"id": "percept_scene",   "family": "perceptual", "template": "He looked at the {w} and"},
    {"id": "relational_kind", "family": "relational", "template": "{w} is a kind of"},
    {"id": "relational_like", "family": "relational", "template": "{w} is similar to"},
]


def frames_for(word: str) -> list[dict]:
    """Return [{id, family, prompt}, ...] for a probe word."""
    return [
        {"id": f["id"], "family": f["family"], "prompt": f["template"].format(w=word)}
        for f in FRAMES
    ]
