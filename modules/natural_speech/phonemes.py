"""Phoneme inventory and acoustic targets (ARPAbet-style subset, ~39 phonemes).

Each voiced phone carries formant targets F1/F2/F3 (Hz) and bandwidths (Hz).
Unvoiced phones carry noise-source definitions (band center/width).
Durations are relative multipliers applied to the base segment length.
"""

VOWELS = {
    "AA": {"f": [730, 1090, 2440], "bw": [80, 100, 140]},
    "AE": {"f": [660, 1720, 2410], "bw": [90, 110, 140]},
    "AH": {"f": [640, 1190, 2390], "bw": [90, 110, 140]},
    "AO": {"f": [570, 840, 2410], "bw": [80, 100, 140]},
    "EH": {"f": [530, 1840, 2480], "bw": [80, 100, 140]},
    "ER": {"f": [490, 1350, 1690], "bw": [80, 110, 160]},
    "IH": {"f": [400, 1950, 2550], "bw": [60, 100, 140]},
    "IY": {"f": [280, 2250, 2890], "bw": [60, 90, 120]},
    "UH": {"f": [450, 1030, 2380], "bw": [70, 100, 140]},
    "UW": {"f": [310, 870, 2250], "bw": [60, 90, 130]},
}

DIPHTHONGS = {
    "AW": {"from": [730, 1090, 2440], "to": [370, 900, 2200]},
    "AY": {"from": [730, 1090, 2440], "to": [300, 2200, 2900]},
    "EY": {"from": [530, 1840, 2480], "to": [300, 2200, 2900]},
    "OW": {"from": [570, 840, 2410], "to": [370, 900, 2200]},
    "OY": {"from": [570, 840, 2410], "to": [300, 2200, 2900]},
}

NASALS = {
    "M": {"f": [250, 1050, 2100], "bw": [80, 180, 260]},
    "N": {"f": [250, 1550, 2550], "bw": [80, 180, 260]},
    "NG": {"f": [250, 2100, 2700], "bw": [80, 220, 300]},
}

LIQUIDS = {
    "L": {"f": [360, 1300, 2800], "bw": [70, 120, 180]},
    "R": {"f": [420, 1300, 1620], "bw": [70, 120, 180]},
}

GLIDES = {
    "W": {"f": [300, 620, 2200], "bw": [60, 90, 140]},
    "Y": {"f": [280, 2200, 2900], "bw": [60, 90, 120]},
}

FRICATIVES = {
    "F": {"noise": (4200, 3800), "level": 0.20},
    "V": {"noise": (4200, 3800), "level": 0.09, "voiced": True},
    "TH": {"noise": (6300, 2800), "level": 0.16},
    "DH": {"noise": (6300, 2800), "level": 0.08, "voiced": True},
    "S": {"noise": (5600, 3000), "level": 0.34, "hiss": True},
    "Z": {"noise": (5600, 3000), "level": 0.16, "voiced": True},
    "SH": {"noise": (2800, 2200), "level": 0.34, "hiss": True},
    "ZH": {"noise": (2800, 2200), "level": 0.16, "voiced": True},
    "HH": {"next_vowel": True},
}

STOPS = {
    "P": {"burst": 900, "bw": 1200, "aspirate": 35, "voiced": False},
    "B": {"burst": 900, "bw": 1200, "aspirate": 0, "voiced": True},
    "T": {"burst": 3400, "bw": 2200, "aspirate": 40, "voiced": False},
    "D": {"burst": 3400, "bw": 2200, "aspirate": 0, "voiced": True},
    "K": {"burst": 2100, "bw": 1600, "aspirate": 45, "voiced": False},
    "G": {"burst": 2100, "bw": 1600, "aspirate": 0, "voiced": True},
}

AFFRICATES = {
    "CH": {"stop": "T", "fric": "SH"},
    "JH": {"stop": "D", "fric": "ZH"},
}

SILENT = {"SIL": {}, "SP": {}, "BRK": {}, "COMMA": {}, "WBRK": {}}

ALL = {}
ALL.update(VOWELS)
ALL.update(DIPHTHONGS)
ALL.update(NASALS)
ALL.update(LIQUIDS)
ALL.update(GLIDES)
ALL.update(FRICATIVES)
ALL.update(STOPS)
ALL.update(AFFRICATES)
ALL.update(SILENT)

VOICED = set(VOWELS) | set(DIPHTHONGS) | set(NASALS) | set(LIQUIDS) | \
    set(GLIDES) | {"V", "DH", "Z", "ZH", "B", "D", "G", "JH"}

DURATION = {
    "vowel": 1.00, "diphthong": 1.45, "nasal": 0.62, "liquid": 0.60,
    "glide": 0.55, "fricative": 0.75, "stop": 0.90, "affricate": 1.35,
    "silent": 1.00,
}

VOWEL_GROUP = set(VOWELS) | set(DIPHTHONGS)


def phone_group(ph):
    for name, table in (("vowel", VOWELS), ("diphthong", DIPHTHONGS),
                        ("nasal", NASALS), ("liquid", LIQUIDS),
                        ("glide", GLIDES), ("fricative", FRICATIVES),
                        ("affricate", AFFRICATES)):
        if ph in table:
            return name
    if ph in STOPS:
        return "stop"
    return "silent"
