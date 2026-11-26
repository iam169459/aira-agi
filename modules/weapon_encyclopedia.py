"""Educational weapons encyclopedia for Aira AGI - facts only.

Scope: weapon CLASSES, HISTORY, REGULATION and arms-control facts suitable
for a school-level or civics-level question. Everything here is public
encyclopedic knowledge.

Hard limit, by design and by policy: NO build, synthesis, modification,
acquisition or usage instructions of any kind for real-world weapons. The
``lookup()`` gate detects instructional intent (how-to, DIY, recipe,
step-by-step phrasing) and refuses those questions with an explanation of
the boundary. The module contains no quantities, procedures or recipes for
harming anyone - ``safety_scan()`` lets the test suite prove that.
"""
import re

REFUSAL = (
    "That's a how-to question, and I don't provide instructions for "
    "building, synthesizing, acquiring or using real weapons of any kind. "
    "What I can do is explain the history, types, technology at a glance "
    "and the laws and treaties around them - ask me for that instead."
)

HOWTO_PATTERNS = (
    r"\bhow\s+(?:to|do\s+i|can\s+i|does\s+one)\s+"
    r"(?:build|make|construct|create|synthesize|manufacture|"
    r"assemble|produce|get|obtain|acquire|use|fire|operate|deploy)",
    r"\b(?:step[- ]by[- ]step|diy|recipe|blueprint|instructions)\b.*"
    r"\b(?:weapon|gun|bomb|explosive|missile)\b",
    r"\b(?:build|make|construct|synthesize|manufacture)\b.*"
    r"\b(?:weapon|gun|bomb|explosive|missile|i ?e ?d)\b",
    r"\b(?:weapon|gun|bomb|explosive|missile)\w*\s+"
    r"(?:build|making|construction|synthesis|manufactur|recipe)",
)
_HOWTO_RE = [re.compile(p, re.IGNORECASE) for p in HOWTO_PATTERNS]


def howto_intent(text):
    """True when the question asks for instructions rather than facts."""
    return any(rx.search(text or "") for rx in _HOWTO_RE)


# --------------------------------------------------------------- classes
CLASSES = {
    "small arms": {
        "era": "14th century CE onward",
        "text": (
            "Portable firearms carried and operated by one person: "
            "muskets, rifles, pistols and submachine guns. Their history "
            "runs from hand cannons through flintlocks to modern "
            "self-loading designs. They dominate small-unit conflict and "
            "are the most widely privately owned weapon class, which is "
            "why most countries regulate them through licensing rather "
            "than prohibition."),
    },
    "artillery": {
        "era": "12th-13th century CE onward",
        "text": (
            "Large-caliber crew-served weapons - cannons, howitzers, "
            "mortars and rocket artillery - that fire projectiles well "
            "beyond direct-sight range. Artillery has been called the "
            "'king of battle' because it caused the majority of combat "
            "casualties in 20th-century wars. Modern systems pair "
            "precise ballistics computation with guided shells."),
    },
    "armored vehicles": {
        "era": "1916 (First World War)",
        "text": (
            "Land combat vehicles protected by armor: tanks, infantry "
            "fighting vehicles and self-propelled guns. The tank first "
            "appeared with the British Mark I in 1916 to break the "
            "stalemate of trench warfare. Armor design and anti-armor "
            "weapons have evolved against each other ever since."),
    },
    "naval vessels": {
        "era": "Antiquity - present",
        "text": (
            "Fighting ships, from ancient galleys and wooden "
            "ships-of-the-line to dreadnought battleships, aircraft "
            "carriers and modern submarines. HMS Dreadnought (1906) made "
            "every earlier battleship obsolete overnight, and submarine "
            "campaigns twice nearly cut Britain's supply lines in the "
            "World Wars. Aircraft carriers replaced battleships as the "
            "capital ship after 1941-1945."),
    },
    "military aircraft": {
        "era": "1914 (First World War)",
        "text": (
            "Airpower platforms: fighters, bombers, attack helicopters "
            "and drones. Early biplanes flew reconnaissance missions in "
            "1914; by 1945 strategic bombing, carrier aviation and jet "
            "propulsion had transformed war. Today crewed aircraft share "
            "the sky with remotely piloted and increasingly autonomous "
            "drones, raising new legal and ethical questions."),
    },
    "missiles": {
        "era": "1944 (Second World War)",
        "text": (
            "Self-propelled guided or unguided munitions: ballistic "
            "missiles, cruise missiles, air-defense and anti-tank "
            "systems. Germany's V-2 (1944) was the first ballistic "
            "missile; the postwar era grew it into intercontinental "
            "rockets that carry most nuclear warheads and, in civilian "
            "descendants, astronauts into space."),
    },
    "nuclear weapons": {
        "era": "1945",
        "text": (
            "Explosive devices drawing energy from nuclear fission or "
            "fusion, first used on Hiroshima and Nagasaki in August "
            "1945. Nine states possess them today. Their singular "
            "destructive scale spawned the most consequential "
            "arms-control regime in history - the Non-Proliferation "
            "Treaty (1968) - and the doctrine of deterrence. Ask me "
            "about NPT, CTBT or deterrence for the framework."),
    },
    "chemical": {
        "era": "modern era: First World War (1915)",
        "text": (
            "Toxic agents used to incapacitate or kill - historically "
            "chlorine, phosgene and mustard gas, first used at scale at "
            "Ypres in 1915 and internationally abhorred since. The 1925 "
            "Geneva Protocol banned use in war; the 1993 Chemical "
            "Weapons Convention banned development and stockpiling and "
            "drove verified destruction of declared stockpiles "
            "worldwide. Their dangers are taught in schools; making or "
            "stockpiling them is a treaty violation and a crime in "
            "nearly every country."),
    },
    "biological": {
        "era": "usage norms from antiquity; program era 1930s-1970s",
        "text": (
            "Pathogens or toxins used as weapons. The 1972 Biological "
            "Weapons Convention was the first treaty to ban an entire "
            "class of weapons outright, but it has no verification "
            "body - a standing weakness in arms control. Historical "
            "state programs were wound down through treaty commitments "
            "rather than battlefield outcomes."),
    },
    "cyber": {
        "era": "2000s onward",
        "text": (
            "Code used to disrupt, spy on or damage computer systems - "
            "the newest weapon class. Stuxnet (disclosed 2010) "
            "physically damaged centrifuges at an Iranian enrichment "
            "plant and is the textbook case of a cyber effect crossing "
            "into the physical world. Attribution is hard, defenses "
            "come first (patching, segmentation, backups), and nations "
            "are still writing the rules. This is the class Aira's "
            "consent-gated defensive tooling belongs to."),
    },
}

# -------------------------------------------------------------- timeline
TIMELINE = [
    ("~9th century CE", "Gunpowder first recorded in Chinese "
     "alchemical texts; within centuries it powers fireworks, rockets "
     "and the world's first firearms."),
    ("1364", "Earliest European illustrated hand cannon - the portable "
     "gun arrives in Europe."),
    ("1503", "Battle of Cerignola: the first battle widely credited as "
     "won by gunpowder small arms."),
    ("1861-1865", "The American Civil War showcases rifles, ironclad "
     "warships and machine-gun precursors - industrial war arrives."),
    ("1884", "Hiram Maxim patents the fully automatic machine gun."),
    ("1915", "Second Battle of Ypres: first large-scale chemical "
     "attack; international revulsion begins immediately."),
    ("1916", "British Mark I tank debuts on the Somme."),
    ("1925", "Geneva Protocol bans chemical and biological weapons in "
     "war (use, not possession)."),
    ("1944-1945", "V-2 ballistic missile flies; nuclear weapons are "
     "used on Hiroshima and Nagasaki - the atomic age opens."),
    ("1952", "First hydrogen bomb test - yields enter the megaton "
     "range."),
    ("1957", "Sputnik launches on a modified military missile - the "
     "rocket becomes a space vehicle too."),
    ("1968", "Nuclear Non-Proliferation Treaty signed."),
    ("1972", "Biological Weapons Convention - the first complete class "
     "ban."),
    ("1987", "INF Treaty eliminates an entire category of "
     "intermediate-range nuclear missiles."),
    ("1993", "Chemical Weapons Convention opens for signature; "
     "verified stockpile destruction begins."),
    ("1997", "Ottawa Treaty bans anti-personnel landmines."),
    ("2010", "Stuxnet disclosed - cyber weapons visibly enter the "
     "physical world."),
    ("2013", "Arms Trade Treaty adopted to regulate the conventional "
     "arms market."),
]

# --------------------------------------------------------------- treaties
TREATIES = {
    "NPT": {
        "name": "Treaty on the Non-Proliferation of Nuclear Weapons",
        "year": 1968, "parties": "191 (the widest of any arms treaty)",
        "summary": ("Grand bargain: non-nuclear states give up nuclear "
                    "weapons; the five original nuclear states commit to "
                    "disarmament negotiations; everyone gains access to "
                    "peaceful nuclear technology under IAEA safeguards. "
                    "Reviewed every five years."),
    },
    "CTBT": {
        "name": "Comprehensive Nuclear-Test-Ban Treaty",
        "year": 1996, "parties": "signed widely, not yet in force",
        "summary": ("Bans all nuclear explosions, everywhere. A global "
                    "sensor network (seismic, hydroacoustic, infrasound, "
                    "radionuclide) can detect even small underground "
                    "tests - one of the great verification achievements."),
    },
    "CWC": {
        "name": "Chemical Weapons Convention",
        "year": 1993, "parties": "193",
        "summary": ("Bans development, production, stockpiling and use "
                    "of chemical weapons and mandates verified "
                    "destruction - over 70,000 declared stockpile tonnes "
                    "destroyed under OPCW inspection."),
    },
    "BWC": {
        "name": "Biological and Toxin Weapons Convention",
        "year": 1972, "parties": "180+",
        "summary": ("The first treaty to ban an entire class of weapons, "
                    "but with no inspection organization - arms-control "
                    "advocates have sought verification protocols ever "
                    "since."),
    },
    "ATT": {
        "name": "Arms Trade Treaty",
        "year": 2013, "parties": "110+",
        "summary": ("Regulates the international trade in conventional "
                    "arms: exporters must assess whether transfers could "
                    "facilitate genocide, war crimes or terrorism."),
    },
    "CCW": {
        "name": "Convention on Certain Conventional Weapons",
        "year": 1980, "parties": "120+",
        "summary": ("Frames rules for weapons deemed to cause "
                    "unnecessary suffering - its protocols cover "
                    "incendiaries, blinding lasers, and via later "
                    "protocols mines and booby traps; today's debates "
                    "about autonomous weapons happen in this forum."),
    },
    "Mine Ban (Ottawa)": {
        "name": "Anti-Personnel Mine Ban Convention",
        "year": 1997, "parties": "160+",
        "summary": ("Complete ban on anti-personnel landmines plus "
                    "stockpile destruction and victim-assistance "
                    "obligations - a model of civil-society-driven arms "
                    "control."),
    },
}

# ------------------------------------------------------- tech at a glance
TECH_AT_A_GLANCE = {
    "what makes a missile ballistic": (
        "It is powered only in the initial phase, then follows an "
        "unchanged flight arc - like a thrown stone - which makes the "
        "trajectory predictable and, in principle, interceptable. "
        "Cruise missiles stay powered and fly like aircraft instead."),
    "tank armor work": (
        "Layers and geometry: steel-composite sandwiches, spaced plates "
        "and angled surfaces force incoming projectiles to spend their "
        "energy crossing multiple barriers. Reactive armor adds tiles "
        "that detonate outward, disrupting the incoming round just "
        "before contact."),
    "sonar echo": (
        "Ships and submarines send sound pulses through water - sound "
        "travels about 1.5 km per second in seawater, far better than "
        "radio does - and timing the returning echo gives range, while "
        "frequency shifts reveal movement. Same physics as a bat's "
        "call."),
    "radar see aircraft": (
        "Radio waves reflect off metal surfaces; the returned signal's "
        "timing gives range and its Doppler shift gives speed. Stealth "
        "design shapes surfaces and materials so the energy is deflected "
        "away from the radar instead of back to it."),
    "mutually assured destruction": (
        "The deterrence logic of the nuclear era: if both sides can "
        "retaliate even after absorbing a first strike (survivable "
        "submarines, hardened silos), neither gains from attacking "
        "first - a grim logic that is precisely why nuclear weapons "
        "have not been used in war since 1945."),
}


def classes():
    return sorted(CLASSES)


def lookup(query):
    """Fact lookup over classes, treaties, timeline and tech notes.

    Returns a factual answer string, the module REFUSAL for how-to
    questions, or a helpful miss message listing what it does cover.
    """
    if howto_intent(query):
        return REFUSAL
    q = (query or "").strip().lower()

    for key, text in TECH_AT_A_GLANCE.items():
        words = [w for w in key.split()
                 if w not in ("what", "is", "a", "the", "does", "why",
                              "how", "see")]
        if words and all(w in q for w in words[-2:]):
            return text

    for key, cls in CLASSES.items():
        head = key.split()[0]
        if head in q or key in q:
            return f"{cls['text']} (in use since: {cls['era']})"

    compact = q.replace("-", "").replace(" ", "")
    for code, tr in TREATIES.items():
        if code.lower().replace(" ", "") in compact \
                or tr["name"].lower() in q:
            return (f"{tr['name']} ({tr['year']}) - {tr['summary']} "
                    f"Parties: {tr['parties']}.")
    if "treaty" in q or "arms control" in q:
        listed = ", ".join(f"{code} ({tr['year']})"
                           for code, tr in TREATIES.items())
        return ("Major arms-control treaties I know: " + listed
                + ". Ask me about any of them by name or acronym.")

    if any(word in q for word in ("history", "timeline", "milestone")):
        lines = [f"{when} - {event}" for when, event in TIMELINE]
        return ("Timeline of weapons technology and control:\n"
                + "\n".join(lines))

    return ("I know weapon classes (small arms, artillery, tanks, ships, "
            "aircraft, missiles, nuclear, chemical, biological, cyber), "
            "arms-control treaties (NPT, CWC, BWC, CTBT, ATT, CCW, "
            "Ottawa), and a history timeline. Ask about one - facts "
            "only; I don't provide build or usage instructions.")


def summary():
    """Compact overview of every class - the 'menu' answer."""
    lines = []
    for name, cls in sorted(CLASSES.items()):
        words = cls["text"].split()
        lines.append(f"- {name} ({cls['era']}): "
                     + " ".join(words[:14]) + "...")
    return ("Weapon classes I can discuss (history, regulation, "
            "technology at a glance):\n" + "\n".join(lines))


def safety_scan():
    """Scan every factual string in the module for instructional red
    flags; returns the found ones (test suite expects none)."""
    flags = ["recipe", "ingredients", "step 1", "step-by-step",
             "mix with", "grams of", "synthesize at", "molar ratio",
             "detonator wiring", "acquire"]
    pieces = [REFUSAL, REFUSAL, REFUSAL]
    pieces += [v for c in CLASSES.values()
               for v in c.values() if isinstance(v, str)]
    pieces += [t for _, t in TIMELINE]
    pieces += [str(v) for tr in TREATIES.values() for v in tr.values()
               if isinstance(v, str)]
    pieces += [str(v) for v in TECH_AT_A_GLANCE.values()]
    pieces += [lookup(c) for c in classes()]
    pieces += [lookup(code) for code in TREATIES]
    blob = " ".join(pieces).lower()
    return [f for f in flags if f in blob]


__all__ = ["REFUSAL", "HOWTO_PATTERNS", "howto_intent", "CLASSES",
           "TIMELINE", "TREATIES", "TECH_AT_A_GLANCE", "classes",
           "lookup", "summary", "safety_scan"]
