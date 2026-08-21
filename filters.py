"""Cheap pre-LLM filters that drop adverts we never want to evaluate.

The main one is the furniture filter: beds, pillows, mattresses, armchairs,
tables, wardrobes, etc. are discarded before spending any LLM tokens.
Terms are normalized (lowercase, umlauts -> ae/oe/ue, ß -> ss) and matched with
word boundaries so "bett" does not match inside an unrelated compound and so
ambiguous bare words (e.g. "bank" -> Powerbank) never fire.
"""
import re

# Non-furniture words that would otherwise match (appliances, etc.)
FURNITURE_NEGATIVE = [
    "kuehlschrank", "gefrierschrank", "gefriertruhe",
]

# German furniture/bedding terms in NORMALIZED form (no umlauts).
# Word-boundary matched, so each compound furniture word is listed explicitly.
FURNITURE_KEYWORDS = [
    # Schlaf / sleeping & bedding
    "bett", "betten", "bettgestell", "bettkasten", "bettsofa", "bettwaesche",
    "bettzeug", "futon", "gaestebett", "matratze", "lattenrost", "topper",
    "kissen", "kopfkissen", "nackenkissen", "polster", "decke", "decken",
    "oberbett", "federkern", "boxspring", "boxspringbett",
    # compound bett- words
    "doppelbett", "stockbett", "etagenbett", "hochbett", "kinderbett",
    "babybett", "gitterbett", "polsterbett", "stockbett",
    # Sitz / seating
    "sofa", "couch", "sessel", "stuhl", "stuehle", "sitzbank", "hocker",
    "liege", "chaiselongue", "sitzgruppe", "sitzgarnitur", "polstergarnitur",
    "eckgarnitur", "ecksofa", "schlafsofa", "schlafcouch", "sitzmoebel",
    "wohnlandschaft", "drehstuhl", "buerostuhl", "buerosessel",
    "kinderstuhl", "hochstuhl", "relaxsessel", "ohrensessel",
    "schaukelstuhl", "fernsehsessel",
    # Tische / tables
    "tisch", "couchtisch", "esstisch", "schreibtisch", "nachttisch",
    "beistelltisch", "wickeltisch", "schminktisch", "gartentisch",
    "balkontisch", "sekretaer", "sideboard", "lowboard", "highboard",
    "tv-board", "tv-moebel", "konsolentisch", "kuechentisch",
    # Schränke / storage
    "schrank", "schraenke", "kommode", "kommoden", "regal", "vitrine",
    "anrichte", "kasten", "kaestchen", "schublade", "rollcontainer",
    "garderobe",
    "kleiderstaender", "kredenz", "buffet", "wandregal", "wandschrank",
    "buecherregal", "wohnwand", "standregal", "haengeregal", "schuhschrank",
    "schuhregal", "kleiderschrank", "eckbank", "geschirrschrank",
    # Küche / kitchen
    "kuechenzeile", "kuechenmoebel", "kuechenschrank", "unterschrank",
    "oberschrank", "haengeschrank", "arbeitsplatte", "kuechenblock",
    # Garten / garden
    "gartenmoebel", "gartenbank", "gartenstuhl", "gartensessel",
    "sonnenliege", "liegestuhl", "balkonmoebel", "terrassenmoebel",
    "gartensitzgruppe",
    # Generic furniture
    "moebel", "moebelstueck", "einrichtung", "einrichtungsgegenstand",
    "moeblierung",
]

_UMLAUT_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})

# After the stem, allow common German plural/declension endings so
# "Kommode" matches "Kommoden", "Tisch" matches "Tische/Tischen", etc.
_PLURAL_SUFFIX = r"(?:e|en|er|es|n|s)?"

# For stems of 5+ chars the risk of appearing inside unrelated words is
# negligible, so we relax the left boundary to catch compounds like
# Doppelbett / Schrankwand / Wohnzimmertisch.  Shorter stems keep
# strict boundaries to avoid false positives (e.g. die Bank → Powerbank
# is avoided because bare "bank" is already excluded from the list).
_COMPILED = []
for term in FURNITURE_KEYWORDS:
    left = r"" if len(term) >= 5 else r"(?<![a-z])"
    _COMPILED.append(
        re.compile(left + re.escape(term) + _PLURAL_SUFFIX + r"(?![a-z])")
    )


def normalize(text: str) -> str:
    return text.lower().translate(_UMLAUT_MAP)


_NEGATIVE_COMPILED = [
    re.compile(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])")
    for term in FURNITURE_NEGATIVE
]


def is_furniture(advert: dict) -> bool:
    """Return True if the advert's title/body looks like furniture/bedding."""
    text = normalize(f"{advert.get('title', '')} {advert.get('body', '')}")
    # Negative list takes precedence — e.g. Kühlschrank is an appliance.
    if any(pattern.search(text) for pattern in _NEGATIVE_COMPILED):
        return False
    return any(pattern.search(text) for pattern in _COMPILED)
