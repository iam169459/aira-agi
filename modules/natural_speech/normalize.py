"""Text normalization: expand numbers, currency, symbols and abbreviations
into speakable words. Pure Python, no external data.
"""
import re

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 20: "twentieth",
    30: "thirtieth", 40: "fortieth", 50: "fiftieth", 60: "sixtieth",
    70: "seventieth", 80: "eightieth", 90: "ninetieth", 100: "hundredth",
    1000: "thousandth",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = ["", " thousand", " million", " billion", " trillion"]


def number_to_words(n):
    """Convert an integer to English words."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + \
            (" " + number_to_words(n % 100) if n % 100 else "")
    chunks = []
    scale_index = 0
    while n > 0:
        part = n % 1000
        if part:
            chunks.append(number_to_words(part) + _SCALES[scale_index])
        n //= 1000
        scale_index += 1
    return " ".join(reversed(chunks))


def _fraction_to_words(frac_text):
    if not frac_text:
        return ""
    digits = re.sub(r"\D", "", frac_text)
    parts = [number_to_words(int(d)) for d in digits]
    return " point " + " ".join(parts)


_ABBREVIATIONS = {
    "mr.": "mister", "mrs.": "missus", "ms.": "miss", "dr.": "doctor",
    "prof.": "professor", "st.": "saint", "vs.": "versus", "etc.": "etcetera",
    "e.g.": "for example", "i.e.": "that is", "approx.": "approximately",
    "min.": "minutes", "max.": "maximum", "hr.": "hours", "km": "kilometers",
    "cm": "centimeters", "mm": "millimeters", "kg": "kilograms",
    "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes",
    "ai": "A I", "ok": "okay", "ok.": "okay",
}

_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)


def _expand_url(match):
    raw = match.group(0)
    raw = raw.rstrip(".,;:!?)")
    body = raw.lower()
    words = []
    if body.startswith("http"):
        words.append("H H T P")
        body = body.split("://", 1)[1] if "://" in body else body[7:]
    body = body.replace("www.", " ", 1)
    body = body.replace(".", " dot ").replace("/", " slash ")
    body = body.replace("-", " dash ").replace("_", " underscore ")
    body = body.replace("@", " at ").replace("=", " equals ")
    words.append(body.strip())
    return " ".join(w for w in words if w)


def _expand_number(match):
    text = match.group(0)
    if re.fullmatch(r"\d+:\d{2}", text):
        hour, minute = text.split(":")
        h = number_to_words(int(hour))
        if minute == "00":
            return h + " o clock"
        return h + " " + number_to_words(int(minute))
    if "." in text:
        whole, frac = text.split(".", 1)
        words = number_to_words(int(whole)) if whole else "zero"
        return words + _fraction_to_words(frac)
    if re.fullmatch(r"\d{4}", text) and 1000 <= int(text) <= 2099:
        n = int(text)
        return number_to_words(n // 100) + " " + \
            number_to_words(n % 100).zfill(2).replace("0", " oh ")
    return number_to_words(int(text))


def _expand_money(match):
    dollars, cents = match.group(1), match.group(2) or ""
    words = []
    if dollars and int(dollars) != 1:
        words.append(number_to_words(int(dollars)) + " dollars")
    elif dollars:
        words.append("one dollar")
    if cents:
        if words:
            words.append("and")
        c = int(cents)
        words.append(number_to_words(c) + (" cent" if c == 1 else " cents"))
    return " ".join(words)


def _expand_percent(match):
    num = match.group(1)
    return num + " percent"


_SYMBOLS = {
    "&": " and ", "@": " at ", "#": " number ", "%": " percent",
    "+": " plus ", "=": " equals ", "<": " less than ", ">": " greater than ",
    "*": " star ", "/": " slash ", "\\": " backslash ", "|": " or ",
    "→": " arrow ", "°": " degrees ",
}

_EMOTICONS = {}


def normalize_text(text):
    """Return text with all non-lexical tokens expanded into English words."""
    text = _URL_RE.sub(_expand_url, text)

    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(r"\b" + re.escape(pattern) + r"\b",
                      replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\$(\d+)(?:\.(\d{1,2}))?", _expand_money, text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)%", _expand_percent, text)
    text = re.sub(r"\b([0-9]+)\s*(?:°|degrees)\s*(?:C|c|F|f)?\b",
                  r"\1 degrees", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b",
                  lambda m: _expand_ordinal_word(m), text)
    text = re.sub(r"\b\d+(?::\d{2})?(?:\.\d+)?\b", _expand_number, text)
    text = re.sub(r"\s+", " ", text.strip())

    out = []
    for ch in text:
        out.append(_SYMBOLS.get(ch, ch))
    text = "".join(out)

    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = re.sub(r"(\w)\1{4,}", r"\1\1", text)
    return text.strip()


def _expand_ordinal_word(match):
    body, suffix = match.group(1), match.group(2)
    base = int(body)
    if base in _ORDINALS:
        return _ORDINALS[base]
    if base == 0:
        return "zeroth"
    last_two = base % 100
    if 11 <= last_two <= 13:
        stem = number_to_words(base)
        return stem + " th"
    last = base % 10
    stem = number_to_words(base)
    if stem.endswith("y"):
        stem = stem[:-1] + "ie"
    if suffix in ("st", "nd", "rd") and (
            (suffix == "st" and last == 1) or
            (suffix == "nd" and last == 2) or
            (suffix == "rd" and last == 3)):
        return stem + suffix
    return stem + "th"
