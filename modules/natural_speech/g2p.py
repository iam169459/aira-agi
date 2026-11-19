"""Rule-based English grapheme-to-phoneme conversion, from scratch.

An ordered longest-context rule walk over the spelling, backed by a lexicon of
irregular/ambiguous words. Phonemes use the ARPAbet-style subset defined in
phonemes.py. No external models, dictionaries or APIs are used: the embedded
lexicon below was authored for this project.
"""
import re

# ---------------------------------------------------------------- lexicon ---
# pronunciation: space separated ARPAbet phonemes. '1' suffix marks stress.
LEXICON = {
    "vocal": "V OW K AH L", "assistant": "AH S IH S T AH N T",
    "assistants": "AH S IH S T AH N T S", "aira": "EH R AH",
    "understand": "AH N D ER S T AE N D",
    "understanding": "AH N D ER S T AE N D IH NG",
    # function words / irregular high frequency words
    "a": "AH", "an": "AE N", "the": "DH AH", "to": "T UW", "too": "T UW",
    "of": "AH V", "do": "D UW", "does": "D AH Z", "done": "D AH N",
    "one": "W AH N", "once": "W AH N S", "two": "T UW", "four": "F AO R",
    "eight": "EY T", "who": "HH UW", "you": "Y UW", "your": "Y AO R",
    "yours": "Y AO R Z", "are": "AA R", "were": "W ER", "was": "W AH Z",
    "come": "K AH M", "some": "S AH M", "one's": "W AH N Z",
    "said": "S EH D", "says": "S EH Z", "have": "HH AE V",
    "give": "G IH V", "gives": "G IH V Z", "given": "G IH V AH N",
    "live": "L IH V", "lives": "L IH V Z", "been": "B IH N",
    "again": "AH G EH N", "against": "AH G EH N S T",
    "friend": "F R EH N D", "friends": "F R EH N D Z",
    "could": "K UH D", "would": "W UH D", "should": "SH UH D",
    "through": "TH R UW", "though": "DH OW", "thought": "TH AO T",
    "taught": "T AO T", "bought": "B AO T", "brought": "B R AO T",
    "caught": "K AO T", "laugh": "L AE F", "laughing": "L AE F IH NG",
    "love": "L AH V", "loves": "L AH V Z", "loved": "L AH V D",
    "above": "AH B AH V", "move": "M UW V", "moved": "M UW V D",
    "prove": "P R UW V", "improve": "IH M P R UW V",
    "eye": "AY", "eyes": "AY Z", "where": "W EH R", "there": "DH EH R",
    "here": "HH IY R", "hear": "HH IY R", "heard": "HH ER D",
    "right": "R AY T", "night": "N AY T", "light": "L AY T",
    "bright": "B R AY T", "might": "M AY T", "sight": "S AY T",
    "high": "HH AY", "buy": "B AY", "by": "B AY", "my": "M AY",
    "why": "W AY", "try": "T R AY", "goodbye": "G UH D B AY",
    "every": "EH V R IY", "everybody": "EH V R IY B AA D IY",
    "water": "W AO T ER", "work": "W ER K", "works": "W ER K S",
    "word": "W ER D", "world": "W ER L D", "what": "W AH T",
    "want": "W AA N T", "wants": "W AA N T S", "walk": "W AO K",
    "talk": "T AO K", "walks": "W AO K S", "talks": "T AO K S",
    "put": "P UH T", "pull": "P UH L", "full": "F UH L", "push": "P UH SH",
    "pushed": "P UH SH T", "book": "B UH K", "look": "L UH K",
    "took": "T UH K", "good": "G UH D", "foot": "F UH T",
    "would've": "W UH D AH V", "yes": "Y EH S", "no": "N OW",
    "okay": "OW K EY", "hi": "HH AY", "hey": "HH EY", "hello": "HH AH L OW",
    "go": "G OW", "goes": "G OW Z", "going": "G OW IH NG",
    "oh": "OW", "on": "AA N", "in": "IH N", "at": "AE T", "as": "AE Z",
    "is": "IH Z", "it": "IH T", "its": "IH T S", "it's": "IH T S",
    "i": "AY", "i'm": "AY M", "i've": "AY V", "i'll": "AY L",
    "i'd": "AY D", "me": "M IY", "he": "HH IY", "she": "SH IY",
    "we": "W IY", "be": "B IY", "be": "B IY", "he's": "HH IY Z",
    "she's": "SH IY Z", "we're": "W IY R", "we'll": "W IY L",
    "they": "DH EY", "they're": "DH EH R", "they've": "DH EY V",
    "they'll": "DH EY L", "their": "DH EH R", "theirs": "DH EH R Z",
    "them": "DH EH M", "then": "DH EH N", "than": "DH AE N",
    "this": "DH IH S", "that": "DH AE T", "that's": "DH AE T S",
    "these": "DH IY Z", "those": "DH OW Z", "with": "W IH DH",
    "without": "W IH DH AW T", "will": "W IH L", "won't": "W OW N T",
    "wouldn't": "W UH D AH N T", "couldn't": "K UH D AH N T",
    "shouldn't": "SH UH D AH N T", "don't": "D OW N T",
    "doesn't": "D AH Z AH N T", "didn't": "D IH D AH N T",
    "isn't": "IH Z AH N T", "aren't": "AA R AH N T",
    "wasn't": "W AH Z AH N T", "weren't": "W ER AH N T",
    "hasn't": "HH AE Z AH N T", "haven't": "HH AE V AH N T",
    "can't": "K AE N T", "can": "K AE N", "cannot": "K AE N AA T",
    "not": "N AA T", "know": "N OW", "known": "N OW N", "knows": "N OW Z",
    "knew": "N UW", "new": "N UW", "now": "N AW", "how": "HH AW",
    "when": "W EH N", "why": "W AY", "which": "W IH CH",
    "our": "AW ER", "ours": "AW ER Z", "out": "AW T",
    "about": "AH B AW T", "around": "ER AW N D",
    "were": "W ER", "we're": "W IY R", "everything": "EH V R IY TH IH NG",
    "something": "S AH M TH IH NG", "nothing": "N AH TH IH NG",
    "anything": "EH N IY TH IH NG", "thing": "TH IH NG",
    "because": "B IH K AO Z", "very": "V EH R IY", "over": "OW V ER",
    "other": "AH DH ER", "others": "AH DH ER Z", "mother": "M AH DH ER",
    "brother": "B R AH DH ER", "another": "AH N AH DH ER",
    "weather": "W EH DH ER", "whether": "W EH DH ER",
    "together": "T AH G EH DH ER", "answer": "AE N S ER",
    "listen": "L IH S AH N", "often": "AO F AH N",
    "island": "AY L AH N D", "busy": "B IH Z IY",
    "business": "B IH Z N IH S", "music": "M Y UW Z IH K",
    "people": "P IY P AH L", "sure": "SH UH R", "only": "OW N L IY",
    "ago": "AH G OW", "away": "AH W EY", "also": "AO L S OW",
    "always": "AO L W EY Z", "already": "AO L R EH D IY",
    "almost": "AO L M OW S T", "among": "AH M AH NG",
    "school": "S K UW L", "chemistry": "K EH M IH S T R IY",
    "christmas": "K R IH S M AH S", "who's": "HH UW Z",
    "two": "T UW", "who've": "HH UW V", "whole": "HH OW L",
    "meant": "M EH N T", "says": "S EH Z", "days": "D EY Z",
    "ways": "W EY Z", "kids": "K IH D Z", "love": "L AH V",
    "name": "N EY M", "time": "T AY M", "like": "L AY K",
    "life": "L AY F", "line": "L AY N", "nice": "N AY S",
    "while": "W AY L", "white": "W AY T", "write": "R AY T",
    "wrote": "R OW T", "written": "R IH T AH N",
    "thank": "TH AE NG K", "thanks": "TH AE NG K S",
    "thankful": "TH AE NG K F AH L", "please": "P L IY Z",
    "sorry": "S AA R IY", "welcome": "W EH L K AH M",
    "help": "HH EH L P", "need": "N IY D", "see": "S IY",
    "seen": "S IY N", "soon": "S UW N", "moon": "M UW N", "food": "F UW D",
    "cool": "K UW L", "room": "R UW M", "use": "Y UW Z",
    "used": "Y UW Z D", "user": "Y UW Z ER", "users": "Y UW Z ER Z",
    "you'd": "Y UW D", "you've": "Y UW V", "one": "W AH N",
    "sure": "SH UH R", "picture": "P IH K CH ER", "nature": "N EY CH ER",
    "future": "F Y UW CH ER", "question": "K W EH S CH AH N",
    "today": "T AH D EY", "tomorrow": "T AH M AA R OW",
    "yesterday": "Y EH S T ER D EY", "morning": "M AO R N IH NG",
    "evening": "IY V N IH NG", "afternoon": "AE F T ER N UW N",
    "monday": "M AH N D EY", "tuesday": "T UW Z D EY",
    "wednesday": "W EH N Z D EY", "thursday": "TH ER Z D EY",
    "friday": "F R AY D EY", "saturday": "S AE T ER D EY",
    "sunday": "S AH N D EY", "january": "JH AE N Y UW EH R IY",
    "february": "F EH B Y UW EH R IY", "march": "M AA R CH",
    "april": "EY P R AH L", "may": "M EY", "june": "JH UW N",
    "july": "JH UW L AY", "august": "AO G AH S T",
    "september": "S EH P T EH M B ER", "october": "AA K T OW B ER",
    "november": "N OW V EH M B ER", "december": "D IH S EH M B ER",
    "hour": "AW ER", "hours": "AW ER Z", "minute": "M IH N AH T",
    "minutes": "M IH N AH T S", "second": "S EH K AH N D",
    "seconds": "S EH K AH N D Z", "week": "W IY K", "year": "Y IY R",
    "years": "Y IY R Z", "month": "M AH N TH", "day": "D EY",
    "great": "G R EY T", "good": "G UH D", "morning": "M AO R N IH NG",
    "run": "R AH N", "running": "R AH N IH NG", "come": "K AH M",
    "coming": "K AH M IH NG", "become": "B IH K AH M",
    "someone": "S AH M W AH N", "everyone": "EH V R IY W AH N",
    "anyone": "EH N IY W AH N", "into": "IH N T UW",
    "onto": "AA N T UW", "many": "M EH N IY", "any": "EH N IY",
    "water": "W AO T ER", "after": "AE F T ER", "better": "B EH T ER",
    "little": "L IH T AH L", "make": "M EY K", "makes": "M EY K S",
    "making": "M EY K IH NG", "take": "T EY K", "takes": "T EY K S",
    "taking": "T EY K IH NG", "give": "G IH V", "gave": "G EY V",
    "find": "F AY N D", "found": "F AW N D", "think": "TH IH NG K",
    "thought": "TH AO T", "things": "TH IH NG Z", "want": "W AA N T",
    "tell": "T EH L", "told": "T OW L D", "ask": "AE S K",
    "asked": "AE S K T", "asking": "AE S K IH NG",
    "sure": "SH UH R", "story": "S T AO R IY", "stories": "S T AO R IY Z",
    "more": "M AO R", "most": "M OW S T", "before": "B IH F AO R",
    "start": "S T AA R T", "started": "S T AA R T IH D",
    "stop": "S T AA P", "stopped": "S T AA P T", "open": "OW P AH N",
    "opened": "OW P AH N D", "close": "K L OW Z", "closed": "K L OW Z D",
    "exit": "EH G Z IH T", "quit": "K W IH T", "email": "IY M EY L",
    "code": "K OW D", "computer": "K AH M P Y UW T ER",
    "phone": "F OW N", "read": "R IY D", "read": "R EH D", "eat": "IY T",
    "speak": "S P IY K", "speaking": "S P IY K IH NG",
    "voice": "V OY S", "speech": "S P IY CH", "sound": "S AW N D",
    "human": "HH Y UW M AH N", "machine": "M AH SH IY N",
    "learn": "L ER N", "learning": "L ER N IH NG",
    "sleep": "S L IY P", "smile": "S M AY L", "heart": "HH AA R T",
    "dream": "D R IY M", "work": "W ER K", "happy": "HH AE P IY",
    "sad": "S AE D", "angry": "AE NG G R IY", "tired": "T AY ER D",
    "excited": "IH K S AY T IH D", "afraid": "AH F R EY D",
    "wonderful": "W AH N D ER F AH L", "beautiful": "B Y UW T AH F AH L",
    "amazing": "AH M EY Z IH NG", "awesome": "AO S AH M",
    "perfect": "P ER F IH K T", "system": "S IH S T AH M",
    "systems": "S IH S T AH M Z", "data": "D EY T AH",
    "model": "M AA D AH L", "memory": "M EH M R IY",
    "language": "L AE NG G W IH JH", "tiny": "T AY N IY",
    "small": "S M AO L", "large": "L AA R JH", "big": "B IH G",
    "from": "F R AH M", "for": "F AO R", "or": "AO R", "so": "S OW",
    "if": "IH F", "but": "B AH T", "and": "AE N D", "am": "AE M",
    "of": "AH V", "off": "AO F", "just": "JH AH S T", "also": "AO L S OW",
    "when": "W EH N", "who": "HH UW", "own": "OW N", "old": "OW L D",
    "cold": "K OW L D", "hold": "HH OW L D", "both": "B OW TH",
    "over": "OW V ER", "under": "AH N D ER", "between": "B IH T W IY N",
    "side": "S AY D", "back": "B AE K", "come": "K AH M",
    "girlish": "G ER L IH SH", "girl": "G ER L", "boy": "B OY",
    "man": "M AE N", "woman": "W UH M AH N", "ai": "EY AY",
    "run": "R AH N", "let": "L EH T", "let's": "L EH T S",
    "get": "G EH T", "got": "G AA T", "give": "G IH V",
    "yeah": "Y EH", "yep": "Y EH P", "nope": "N OW P", "um": "AH M",
    "uh": "AH", "hmm": "HH M", "hm": "HH M", "er": "ER", "ah": "AA",
    "mhm": "M HH M", "uh-huh": "AH HH AH", "mm-hmm": "M HH M",
}

VOWEL_LETTERS = set("aeiouy")

_STRESS = "1"


def _phonemes_from_pron(pron):
    out = []
    for tok in pron.split():
        out.append(tok.rstrip(_STRESS))
    return out


def pronunciations():
    """Return a copy of the lexicon mapping spelling -> phoneme list."""
    return {w: _phonemes_from_pron(p) for w, p in LEXICON.items()}


def add_word(spelling, pron):
    """Register a custom word pronunciation, e.g. add_word('aira', 'EH R AH')"""
    LEXICON[spelling.lower().strip()] = pron.lower().strip()


# ------------------------------------------------------------------ rules ---
def _is_vowel(ch):
    return ch in VOWEL_LETTERS


_LONG_OF = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}
_SHORT_OF = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}
_R_VOWEL = {"a": "AA", "e": "EH", "i": "ER", "o": "AO", "u": "ER"}
_R_E = {"a": ("EH", "R"), "e": ("EH", "R"), "i": ("AY", "ER"),
        "o": ("AO", "R"), "u": ("UH", "R"), "y": ("AY", "ER")}

_CONSONANT_CLUSTERS = {
    "sh": ["SH"], "ch": ["CH"], "th": ["TH"], "ph": ["F"], "wh": ["W"],
    "ck": ["K"], "qu": ["K", "W"], "sk": [], "sc": [],
}

_DOUBLE_CONSONANTS = set("bdfglmnpprrssttz")


def _suffix_split(word):
    """Split common suffixes; returns (stem, suffix_kind) or (word, None)."""
    for suf, kind in (("ing", "ing"), ("ies", "ies"), ("ness", "ness"),
                      ("ment", "ment"), ("ly", "ly"), ("er", "er"),
                      ("est", "est"), ("ed", "ed"), ("es", "es"),
                      ("s", "s"), ("y", "y")):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            stem = word[: -len(suf)]
            if suf in ("ed", "es", "er", "est", "ies", "ing") \
                    and stem.endswith("e") and not stem.endswith("ee"):
                stem = stem  # magic-e stems keep their e ("hop(e)ed")
                return stem, ("e_" + kind)
            if kind in ("s", "es") and len(stem) < 2:
                continue
            return stem, kind
    return word, None


def _last_vowel_index(letters):
    for i in range(len(letters) - 1, -1, -1):
        if _is_vowel(letters[i]):
            return i
    return -1


def _stem_phonemes(stem):
    return _spell_to_phonemes(stem)


def _g2p_with_suffix(word):
    stem, suf = _suffix_split(word)
    base = _stem_phonemes(stem) if stem else []
    if suf is None or not base:
        return _spell_to_phonemes(word)
    last = base[-1] if base else ""
    if suf in ("s", "es", "ies"):
        if last in ("S", "Z", "SH", "ZH", "CH", "JH"):
            return base + ["IH", "Z"]
        if last in " ":  # unknown
            return base + ["IH", "Z"]
        stembase = _stem_phonemes(stem[:-1] if suf == "es" and
                                  stem[-1:] not in ("s", "z", "x", "h") else stem)
        base2 = stembase or base
        last = base2[-1]
        voiced = last in ("B", "D", "G", "V", "DH", "Z", "ZH", "M", "N",
                          "NG", "L", "R", "W", "Y") or \
            last in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                     "IH", "IY", "OW", "OY", "UH", "UW")
        if suf == "ies":
            return stembase + ["IY", "Z"]
        if suf == "es" and _stem_phonemes(stem) != stembase:
            return stembase + ["IH", "Z"]
        return base2 + (["Z"] if voiced else ["S"])
    if suf == "ed":
        if last in ("T", "D"):
            return base + ["IH", "D"]
        return base + (["D"] if last in VOICED_ENDS else ["T"])
    if suf == "y":
        return base + ["IY"]
    if suf == "ies":
        return base + ["IY", "Z"]
    if suf == "er":
        return base + ["ER"]
    if suf == "est":
        return base + ["IH", "S", "T"]
    if suf == "ing":
        if stem.endswith(("e",)) and stem[-2:-1] not in VOWEL_LETTERS and \
                not stem.endswith("ee"):
            base = _stem_phonemes(stem[:-1])
        if base and not base[-1].startswith(("AE", "EH", "IH", "AA")):
            pass
        return base + ["IH", "NG"]
    if suf == "ly":
        return base + ["L", "IY"]
    if suf == "ness":
        return base + ["N", "AH", "S"]
    if suf == "ment":
        return base + ["M", "EH", "N", "T"]
    if suf == _STRESS:
        return base
    return _spell_to_phonemes(word)


VOICED_ENDS = set("BVGDVLMNRWYZHJ") | {
    "B", "D", "G", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "R", "W", "Y"}
VOICED_ENDS = {"B", "D", "G", "V", "DH", "Z", "ZH", "M", "N", "NG", "L",
               "R", "W", "Y"}


def _has_vowel(word):
    return any(c in VOWEL_LETTERS for c in word)


def _spell_to_phonemes(w):
    out = []
    i = 0
    n = len(w)
    while i < n:
        c = w[i]
        c2 = w[i:i + 2]
        c3 = w[i:i + 3]
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""
        prev = w[i - 1] if i > 0 else ""
        last = (i == n - 1)

        # --- 3-char patterns ------------------------------------------------
        if c3 == "igh":
            out.append("AY")
            i += 3
            continue
        if c3 == "tch":
            out.append("CH")
            i += 3
            continue
        if c3 == "dge":
            out.append("JH")
            i += 3
            continue
        if c3 == "sch":
            out.extend(["S", "K"])
            i += 3
            continue
        if c3 == "eau":
            out.append("OW")
            i += 3
            continue
        if c3 == "tion":
            out.extend(["SH", "AH", "N"])
            i += 3
            continue
        if c3 == "ture":
            out.extend(["CH", "ER"])
            i += 3
            continue
        if c3 == "ssion":
            out.extend(["SH", "AH", "N"])
            i += 4
            continue
        if c3 == "ssio":
            out.extend(["SH", "AH"])
            i += 3
            continue
        if c3 == "ough":
            out.append("OW")
            i += 3
            continue
        if c3 == "augh":
            out.extend(["AA", "F"])
            i += 3
            continue
        if c3 == "sion":
            out.extend(["ZH", "AH", "N"])
            i += 3
            continue
        if c3 == "air" or (c2 == "ar" and nxt2 == "e" and i + 3 == n):
            out.extend(["EH", "R"])
            i += 2
            continue
        if c2 in ("ir", "er") and nxt2 == "e" and i + 3 == n:
            out.extend(["ER"])
            i += 2
            continue

        # --- 2-char patterns ------------------------------------------------
        if c2 == "gh":
            if i == 0 and nxt:
                out.append("G")
                i += 2
                continue
            i += 2
            continue
        if c2 == "ai":
            out.append("EY")
            i += 2
            continue
        if c2 == "ay":
            out.append("EY")
            i += 2
            continue
        if c2 == "au":
            out.append("AO")
            i += 2
            continue
        if c2 == "aw":
            out.append("AO")
            i += 2
            continue
        if c2 == "ea":
            if nxt2 == "r":
                out.extend(["EH", "R"])
                i += 3
                continue
            out.append("IY")
            i += 2
            continue
        if c2 == "ee":
            out.append("IY")
            i += 2
            continue
        if c2 == "ei":
            out.append("EY")
            i += 2
            continue
        if c2 == "ey":
            out.append("IY" if nxt == "" else "EY")
            i += 2
            continue
        if c2 == "eu" or c2 == "ew":
            out.append("UW")
            i += 2
            continue
        if c2 == "ie":
            if last and i == n - 2:
                out.append("AY")
            else:
                out.append("IY")
            i += 2
            continue
        if c2 == "oa":
            out.append("OW")
            i += 2
            continue
        if c2 == "oe":
            out.append("OW")
            i += 2
            continue
        if c2 == "oi" or c2 == "oy":
            out.append("OY")
            i += 2
            continue
        if c2 == "oo":
            out.append("UW")
            i += 2
            continue
        if c2 == "ou":
            out.append("AW")
            i += 2
            continue
        if c2 == "ue" or c2 == "ui":
            out.append("UW")
            i += 2
            continue
        if c2 == "ng":
            if nxt == "e" and (i + 3 >= n):
                out.extend(["N", "JH"])
                i += 3
                continue
            out.append("NG")
            i += 2
            continue
        if c2 == "nk":
            out.extend(["NG", "K"])
            i += 2
            continue
        if c2 == "kn" and i == 0:
            out.append("N")
            i += 2
            continue
        if c2 == "wr" and i == 0:
            out.append("R")
            i += 2
            continue
        if c2 == "gn" and i == 0:
            out.append("N")
            i += 2
            continue
        if c2 == "ch":
            out.append("CH")
            i += 2
            continue

        # --- single letters -------------------------------------------------
        if c == "a" or c == "e" or c == "i" or c == "o" or c == "u":
            out.extend(_vowel_rule(w, i))
            i = _vowel_advance(w, i, out)
            continue
        if c == "y":
            if i == 0:
                out.append("Y")
            elif nxt and not _is_vowel(nxt) and _has_vowel(w[i + 1:]):
                out.append("IH")
            elif last:
                out.append("IY" if prev else "AY")
            else:
                out.append("AY" if not _has_vowel(w[:i]) + w[i + 1:] else "IH")
            i += 1
            continue
        if c == "c":
            if nxt == "c":
                if nxt2 in "eiy":
                    out.extend(["K", "S"])
                    i += 2
                    continue
                out.append("K")
                i += 2
                continue
            if nxt in "eiy":
                out.append("S")
            else:
                out.append("K")
            i += 1
            continue
        if c == "g":
            if nxt == "g":
                out.append("G")
                i += 2
                continue
            if nxt in "eiy":
                if nxt2 == "g":  # "suggest" style
                    out.append("G")
                else:
                    out.append("JH")
            else:
                out.append("G")
            i += 1
            continue
        if c == "s":
            if nxt == "s":
                out.append("S")
                i += 2
                continue
            if nxt == "h":
                out.append("SH")
                i += 2
                continue
            _p = prev if prev else ""
            if _p and _is_vowel(_p) and nxt and _is_vowel(nxt):
                out.append("Z")
            else:
                out.append("S")
            i += 1
            continue
        if c == "x":
            out.extend(["G", "Z"] if (prev and nxt) else ["K", "S"])
            i += 1
            continue
        if c == "j":
            out.append("JH")
            i += 1
            continue
        if c == "z":
            out.append("Z")
            i += 1
            continue
        if c == "q":
            out.append("K")
            i += 1
            continue
        if c == "h":
            out.append("HH")
            i += 1
            continue
        if c == "f":
            out.append("F" if nxt != "f" else "F")
            i += 2 if nxt == "f" else 1
            continue
        if c == "b":
            out.append("B")
            i += 2 if nxt == "b" else 1
            continue
        if c == "v":
            out.append("V")
            i += 1
            continue
        if c == "w":
            out.append("W")
            i += 1
            continue
        if c == "k":
            out.append("K")
            i += 2 if nxt == "k" else 1
            continue
        if c == "t":
            out.append("T")
            i += 2 if nxt == "t" else 1
            continue
        if c == "d":
            out.append("D")
            i += 2 if nxt == "d" else 1
            continue
        if c == "p":
            out.append("P")
            i += 2 if nxt == "p" else 1
            continue
        if c == "m":
            out.append("M")
            i += 2 if nxt == "m" else 1
            continue
        if c == "n":
            out.append("N")
            i += 2 if nxt == "n" else 1
            continue
        if c == "r":
            out.append("R")
            i += 2 if nxt == "r" else 1
            continue
        if c == "l":
            out.append("L")
            if nxt == "l":
                i += 2
            else:
                i += 1
            continue
        # unknown char: skip
        i += 1
    return _cleanup(out)


def _cleanup(phonemes):
    """Collapse obvious artifacts from the rule walk."""
    out = []
    for p in phonemes:
        out.append(p)
    merged = []
    for p in out:
        if merged and merged[-1] == "AH" and p == "L" and len(merged) >= 2 \
                and merged[-2] in ("T", "D", "K", "B", "P", "G", "F", "M"):
            merged[-1] = "L"
        else:
            merged.append(p)
    return merged


def _vowel_rule(w, i):
    """Phonemes for the single vowel at position i of word w."""
    c = w[i]
    n = len(w)
    nxt = w[i + 1] if i + 1 < n else ""
    nxt2 = w[i + 2] if i + 2 < n else ""
    after = w[i + 3] if i + 3 < n else ""

    if nxt == "r":
        if i + 3 == n and nxt2 == "e":
            f1, r = _R_E[c]
            return [f1, r]
        comp = {"a": "AA", "e": "ER", "i": "ER", "o": "AO", "u": "ER"}[c]
        return [comp, "R"]
    if c == "u" and nxt == "r":
        return ["ER", "R"]

    if nxt == "l" and nxt2 and _is_vowel(nxt2):
        return {"a": "AO", "e": "EH", "o": "OW", "i": "IH", "u": "UH"}[c] if \
            c != "a" else ["AO", "L"]

    if i == 0 and c == "o" and nxt == "n" and nxt2 == "e" and i + 3 == n - 1:
        pass

    if nxt and nxt not in VOWEL_LETTERS:
        if nxt2 == "" and nxt == "e" and (w[-1] == "e"):
            return [_LONG_OF[c]]
        if nxt2 and not _is_vowel(nxt2) and (nxt2 == "e" and i + 3 == n):
            return [_LONG_OF[c]]
        if nxt2 and _is_vowel(nxt2):
            if c == "e" and nxt == "r":
                return ["ER", "R"]
            if nxt in "x":
                return [_SHORT_OF[c]]
            if nxt2 == "y" and i + 3 == n:
                return [_SHORT_OF[c]]
            if nxt == "l" and c == "a":
                return ["AO", "L"]
            return [_LONG_OF[c]]
        if nxt == "l" and nxt2 == "l" and i + 3 <= n:
            if c == "a":
                return ["AO", "L"]
        return [_SHORT_OF[c]]
    if nxt == "":
        return [_SHORT_OF[c]]
    return [_SHORT_OF[c]]


def _vowel_advance(w, i, out):
    """Advance index past the vowel and any consonant letters consumed."""
    c = w[i]
    n = len(w)
    nxt = w[i + 1] if i + 1 < n else ""
    nxt2 = w[i + 2] if i + 2 < n else ""
    if nxt == "r":
        return i + 2
    if nxt == "l" and nxt2 and _is_vowel(nxt2):
        if c == "a":
            return i + 1
    return i + 1


def _possessive_or_contraction(word):
    if word.endswith("'s"):
        base = _g2p_word_raw(word[:-2])
        last = base[-1] if base else "S"
        voiced = last in ("B", "D", "G", "V", "DH", "Z", "ZH", "M", "N",
                          "NG", "L", "R", "W", "Y") or last in (
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
            "IY", "OW", "OY", "UH", "UW")
        return base + (["Z"] if voiced else ["S"])
    if word.endswith("n't"):
        lex = LEXICON.get(word)
        if lex:
            return _phonemes_from_pron(lex)
    return None


def _g2p_word_raw(word):
    word = word.lower().strip()
    lex = LEXICON.get(word)
    if lex:
        return _phonemes_from_pron(lex)
    stripped = re.sub(r"[^a-z']", "", word)
    if not stripped:
        return []
    if "'" in stripped:
        got = _possessive_or_contraction(stripped)
        if got:
            return got
        parts = stripped.split("'")
        if word == "wo":
            pass
        out = []
        for p in parts:
            out.extend(_g2p_word_raw(p) if p not in LEXICON
                       else _phonemes_from_pron(LEXICON[p]))
            if out and out[-1] != "BRK":
                out.append("BRK")
        return [p for p in out if p != "BRK"]
    if not _has_vowel(stripped):
        return list(stripped.upper())
    return _g2p_with_suffix(stripped)


_WORD_RE = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9\s]")


def g2p_word(word):
    """Convert one word (with punctuation allowed) to phonemes."""
    return _g2p_word_raw(word)


def g2p(text):
    """Convert text to a phoneme list.

    Punctuation becomes SIL (long pause) or COMMA (short pause) markers.
    Returns (phonemes, word_count).
    """
    words = _WORD_RE.findall(text)
    phonemes = []
    wc = 0
    for tok in words:
        if tok.isspace():
            continue
        if re.match(r"^[A-Za-z0-9']+$", tok):
            ph = _g2p_word_raw(tok)
            if ph:
                phonemes.extend(ph)
                phonemes.append("WBRK")
                wc += 1
        else:
            if tok in ",;:":
                phonemes.append("COMMA")
            elif tok in ".!?":
                phonemes.append("SIL")
            elif tok == "-":
                phonemes.append("WBRK")
    if phonemes and phonemes[0] == "WBRK":
        phonemes = phonemes[1:]
    return phonemes, wc
