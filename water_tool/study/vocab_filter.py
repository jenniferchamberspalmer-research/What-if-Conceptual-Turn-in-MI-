"""English-only vocabulary filter — the study's PRIMARY CONTROL (design Section 3).

The multilingual-shared-space rebuttal (Section 3) says the cross-linguistic
neighbors of *water* are just an artifact of a pooled multilingual embedding
space, not evidence of *langue*. The decisive answer is to re-run the neighbor
analysis restricted to a SINGLE language (English) and ask whether the
same-language neighbors are still systemic (liquid/fluid/moisture) rather than
experiential (drink/thirst/splash).

That requires tagging which of Gemma's ~256k vocabulary tokens are English.
The hard part is NOT script: *agua*, *Wasser*, *eau*, *casa* are all Latin
script. A script-only filter would let them through and silently defeat the
control. So a token counts as English only if its decoded string is

  1. pure ASCII alphabetic (drops 水 / воды / الماء / punctuation / digits /
     subword fragments with symbols, and incidentally accented Latin), AND
  2. a real, reasonably-common English word per `wordfreq`, AND
  3. NOT dominated by another Latin-script language (English Zipf within
     `dominance_margin` of the top competing language's Zipf).

Rule 3 is what rejects *agua* (Spanish-dominant) and *Wasser* (German-dominant)
while keeping genuine English words that happen to be cognates (*no*, *final*,
*animal*). Thresholds are parameters and are chosen by the validation pass
(`validate`) against a held-out labeled sample, per design Section 10.
"""

from functools import lru_cache

# Latin-script languages that compete with English for ASCII strings.
# Intersected with wordfreq's available languages at call time.
_COMPETITOR_LANGS = [
    "es", "fr", "de", "it", "pt", "nl", "sv", "nb", "da",
    "pl", "cs", "ro", "tr", "id", "ms", "ca", "fi", "hu", "vi",
]

DEFAULT_MIN_EN_ZIPF = 2.2      # ~ >=1.6 occurrences per 10M words: drops junk/fragments
DEFAULT_DOMINANCE_MARGIN = 0.9  # reject if a competitor language outscores English by more


def _competitor_langs():
    from wordfreq import available_languages
    avail = set(available_languages().keys())
    return [l for l in _COMPETITOR_LANGS if l in avail]


@lru_cache(maxsize=200_000)
def _scores(word_lower: str) -> tuple[float, float]:
    """(english_zipf, max_competitor_zipf) for a lowercased word."""
    from wordfreq import zipf_frequency
    en = zipf_frequency(word_lower, "en")
    others = max((zipf_frequency(word_lower, l) for l in _competitor_langs()),
                 default=0.0)
    return en, others


def is_english(
    s: str,
    min_en_zipf: float = DEFAULT_MIN_EN_ZIPF,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> bool:
    """Decide whether a decoded token string is an English word.

    `s` should already be stripped of surrounding whitespace.
    """
    if not s or not s.isascii() or not s.isalpha():
        return False
    en, others = _scores(s.lower())
    if en < min_en_zipf:
        return False
    return en >= others - dominance_margin


def guess_language(s: str) -> str:
    """Coarse language label for a decoded token, for reporting the
    multilingual set (design Section 6: 'record ... language')."""
    if not s:
        return "empty"
    # Script first — non-Latin scripts are unambiguous.
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            return "zh/ja(han)"
        if 0x3040 <= o <= 0x30FF:
            return "ja(kana)"
        if 0x0400 <= o <= 0x04FF:
            return "ru/cyrillic"
        if 0x0600 <= o <= 0x06FF:
            return "ar/fa"
        if 0x0590 <= o <= 0x05FF:
            return "he"
        if 0x0900 <= o <= 0x097F:
            return "hi/deva"
        if 0x0E00 <= o <= 0x0E7F:
            return "th"
        if 0xAC00 <= o <= 0xD7A3:
            return "ko"
    if not s.isascii() or not s.isalpha():
        return "other/symbol"
    # Latin script: argmax over English + competitors via wordfreq.
    from wordfreq import zipf_frequency
    wl = s.lower()
    scores = {"en": zipf_frequency(wl, "en")}
    for l in _competitor_langs():
        scores[l] = zipf_frequency(wl, l)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "latin/unknown"


def build_english_ids(
    tok,
    min_en_zipf: float = DEFAULT_MIN_EN_ZIPF,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> dict:
    """Scan the whole vocabulary, return English token ids + summary stats.

    Returns a dict with keys: english_ids (sorted list[int]), vocab_size,
    n_english, thresholds, and a small sample of accepted/rejected decodes.
    """
    vocab_size = len(tok)
    english_ids = []
    accepted_sample, rejected_latin_sample = [], []
    for idx in range(vocab_size):
        s = tok.decode([idx]).strip()
        if is_english(s, min_en_zipf, dominance_margin):
            english_ids.append(idx)
            if len(accepted_sample) < 40:
                accepted_sample.append(s)
        elif s.isascii() and s.isalpha() and len(s) > 2 and len(rejected_latin_sample) < 40:
            # Latin-script words we rejected — useful to eyeball for over-rejection.
            rejected_latin_sample.append(s)
    return {
        "english_ids": english_ids,
        "vocab_size": vocab_size,
        "n_english": len(english_ids),
        "thresholds": {"min_en_zipf": min_en_zipf,
                       "dominance_margin": dominance_margin},
        "accepted_sample": accepted_sample,
        "rejected_latin_sample": rejected_latin_sample,
    }


# ---- Validation (design Section 10: validate against held-out labeled tokens) ----

# Real English words the filter MUST accept.
_KNOWN_ENGLISH = [
    "water", "liquid", "fluid", "moisture", "vapor", "stone", "rock",
    "bread", "loaf", "time", "justice", "truth", "number", "pain", "warmth",
    "hunger", "thirst", "here", "now", "this", "freedom", "honor", "home",
    "hand", "fire", "blood", "tree", "milk", "bone", "drink", "wet", "river",
    "ocean", "ice", "steam", "salt", "soul", "mercy", "today",
]

# Latin-script NON-English words the filter MUST reject (the hard cases).
_KNOWN_FOREIGN_LATIN = [
    "agua", "wasser", "eau", "casa", "maison", "haus", "acqua", "vatten",
    "woda", "voda", "vand", "vesi", "fuego", "feu", "feuer", "pomme",
    "queso", "leite", "fuoco", "wody",
]

# Non-Latin tokens the filter MUST reject (handled by the script gate).
_KNOWN_NONLATIN = ["水", "воды", "الماء", "पानी", "מים", "น้ำ"]


def validate(
    min_en_zipf: float = DEFAULT_MIN_EN_ZIPF,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> dict:
    """Run the filter on labeled samples and return precision/recall + misses."""
    def acc(words):
        return [(w, is_english(w.strip(), min_en_zipf, dominance_margin)) for w in words]

    en = acc(_KNOWN_ENGLISH)
    foreign = acc(_KNOWN_FOREIGN_LATIN)
    nonlatin = acc(_KNOWN_NONLATIN)

    tp = sum(1 for _, v in en if v)            # English correctly kept
    fn = [w for w, v in en if not v]           # English wrongly dropped
    fp = [w for w, v in foreign + nonlatin if v]  # non-English wrongly kept
    tn = sum(1 for _, v in foreign + nonlatin if not v)

    n_pos = len(en)
    n_neg = len(foreign) + len(nonlatin)
    recall = tp / n_pos if n_pos else 0.0
    # Precision over the labeled set: kept-correct / total-kept.
    kept = tp + len(fp)
    precision = tp / kept if kept else 1.0
    specificity = tn / n_neg if n_neg else 0.0
    return {
        "thresholds": {"min_en_zipf": min_en_zipf, "dominance_margin": dominance_margin},
        "recall_english": round(recall, 3),
        "precision": round(precision, 3),
        "specificity_nonenglish": round(specificity, 3),
        "english_dropped (false negatives)": fn,
        "nonenglish_kept (false positives)": fp,
        "n_english_labeled": n_pos,
        "n_nonenglish_labeled": n_neg,
    }
