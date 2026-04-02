"""
nlp_parser.py
-------------
Extracts city, state, and locality from a cleaned address string.

TECHNIQUE: Fuzzy String Matching
Measures string similarity (0-100%) instead of exact matching.
Handles typos, abbreviations, and colloquial city names.

Libraries: rapidfuzz (pip install rapidfuzz) or difflib (built-in)
"""

import re

try:
    from rapidfuzz import process as fuzz_process, fuzz
    RAPIDFUZZ = True
except ImportError:
    import difflib
    RAPIDFUZZ = False

from pincode_lookup import get_all_cities, get_all_states


# ---------------------------------------------------------------
# CITY ALIAS MAP
# ---------------------------------------------------------------
# Two purposes:
#
# 1. Common name → official dataset name
#    "bangalore" → "bengaluru urban"  (official spelling differs)
#    "gurgaon"   → "gurugram"         (city was renamed)
#
# 2. Direct city names → themselves
#    "chennai" → "chennai"
#    WHY: alias check scores 100% and always wins over fuzzy matches.
#    Without this, a false fuzzy match earlier in the token list
#    would fire before we even reach the correct city token.

CITY_ALIASES = {
    # common name → official dataset name
    "delhi":      "new delhi",
    "dilli":      "new delhi",
    "bangalore":  "bengaluru urban",
    "bangaluru":  "bengaluru urban",
    "bengaluru":  "bengaluru urban",
    "bombay":     "mumbai",
    "madras":     "chennai",
    "calcutta":   "kolkata",
    "gurgaon":    "gurugram",
    "noida":      "gautam buddha nagar",
    "ghaziabad":  "ghaziabad",
    # direct names → themselves (fast-path, prevents false fuzzy matches)
    "chennai":    "chennai",
    "mumbai":     "mumbai",
    "kolkata":    "kolkata",
    "hyderabad":  "hyderabad",
    "pune":       "pune",
    "jaipur":     "jaipur",
    "lucknow":    "lucknow",
    "surat":      "surat",
    "patna":      "patna",
    "bhopal":     "bhopal",
    "indore":     "indore",
    "nagpur":     "nagpur",
    "chandigarh": "chandigarh",
    "dehradun":   "dehradun",
    "gurugram":   "gurugram",
    "ahmedabad":  "ahmedabad",
    "coimbatore": "coimbatore",
    "kochi":      "ernakulam",
    "vizag":      "visakhapatnam",
}


# ---------------------------------------------------------------
# CACHED LISTS
# ---------------------------------------------------------------
# Loaded once from the pincode DB, reused on every call.
# Avoids re-reading 165K rows for every address parsed.

_CITY_LIST  = None
_STATE_LIST = None

def _get_city_list():
    global _CITY_LIST
    if _CITY_LIST is None:
        _CITY_LIST = [c.lower() for c in get_all_cities()]
    return _CITY_LIST

def _get_state_list():
    global _STATE_LIST
    if _STATE_LIST is None:
        _STATE_LIST = [s.lower() for s in get_all_states()]
    return _STATE_LIST


# ---------------------------------------------------------------
# FUZZY MATCH WITH SCORE
# ---------------------------------------------------------------
# Returns (best_match, score) or (None, 0).
#
# WHY SequenceMatcher directly instead of get_close_matches:
# get_close_matches uses strictly GREATER THAN internally (score > cutoff)
# so a score of exactly 0.800 at cutoff=0.80 returns nothing.
#
# SequenceMatcher lets us use >= (greater than OR equal to)
# so boundary scores like "dlehi" vs "delhi" = 0.800 pass correctly.
#
# We also need the actual score (not just pass/fail) so the
# sliding window can compare candidates and pick the best one.

def fuzzy_match_with_score(query, choices, threshold=78):
    """
    Find best matching string from choices for query.

    Args:
        query     : word or phrase to match
        choices   : list of known city/state names
        threshold : minimum similarity % to accept (0-100), uses >=

    Returns:
        (best_match, score) or (None, 0)
    """
    if not query or not choices or len(query) < 5:
        return None, 0

    if RAPIDFUZZ:
        result = fuzz_process.extractOne(
            query, choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )
        if result:
            return result[0], result[1]
        return None, 0
    else:
        # Iterate and use >= so boundary scores pass
        best_match = None
        best_score = 0
        for choice in choices:
            score = difflib.SequenceMatcher(None, query, choice).ratio() * 100
            if score >= threshold and score > best_score:
                best_match = choice
                best_score = score
        return best_match, best_score


# ---------------------------------------------------------------
# ALIAS LOOKUP (with fuzzy fallback for typos)
# ---------------------------------------------------------------
# Step 1: exact lookup  — "bangalore"  → found directly
# Step 2: fuzzy lookup  — "bangalor"   → fuzzy matches "bangalore"
#                          → looks up "bangalore" → "bengaluru urban"
#
# Threshold=87 is strict here — alias fuzzy matching is only
# for clear typos of well-known names, not loose approximations.

def alias_lookup(candidate):
    """
    Returns official district name from alias map, or None.
    Tries exact match first, then fuzzy match against alias keys.
    """
    if candidate in CITY_ALIASES:
        return CITY_ALIASES[candidate]

    alias_keys = list(CITY_ALIASES.keys())
    matched_key, score = fuzzy_match_with_score(candidate, alias_keys, threshold=87)
    if matched_key:
        return CITY_ALIASES[matched_key]

    return None


# ---------------------------------------------------------------
# SLIDING WINDOW MATCH — picks BEST match, not just FIRST
# ---------------------------------------------------------------
# Tries largest window first (3 words → 2 words → 1 word).
# At each window size, scans ALL positions and tracks scores.
# Returns the match with the HIGHEST score at that window size.
#
# WHY BEST NOT FIRST:
# Tokens: ["42", "anna", "nagar", "chennai"]
# Old (first): "anna" → "panna" at 88% → returns immediately ✗
# New (best):  "anna"→88%, "nagar"→86%, "chennai"→100% → returns chennai ✓
#
# Alias hits score 100 so they always beat fuzzy matches.

def sliding_window_match(tokens, choices, max_window=3, threshold=78, single_word_threshold=None):
    """
    Find best-scoring match across all token window combinations.
    Returns (matched_name, start_idx, end_idx) or (None, -1, -1)
    """
    for window_size in range(min(max_window, len(tokens)), 0, -1):
        best_match = None
        best_score = 0
        best_pos   = (-1, -1)

        for i in range(len(tokens) - window_size + 1):
            candidate = " ".join(tokens[i : i + window_size])

            # Alias check scores 100 — always beats any fuzzy match
            official = alias_lookup(candidate)
            if official and official in choices:
                if 100 > best_score:
                    best_match = official
                    best_score = 100
                    best_pos   = (i, i + window_size)
                continue

            if len(candidate) < 5:
                continue

            # Single word: stricter threshold (85) to avoid
            # short words like "anna" matching "panna"
            # Multi-word: slightly relaxed (82)
            # single_word_threshold overrides default when caller needs different sensitivity
            # states (36 items): 78 is safe  — 'dlehi'→'delhi' passes, no false matches
            # cities (749 items): 85 needed — prevents 'anna'→'panna' false match
            if single_word_threshold is not None:
                t = single_word_threshold if window_size == 1 else 82
            else:
                t = 85 if window_size == 1 else 82
            match, score = fuzzy_match_with_score(candidate, choices, t)
            if match and score > best_score:
                best_match = match
                best_score = score
                best_pos   = (i, i + window_size)

        if best_match:
            return best_match, best_pos[0], best_pos[1]

    return None, -1, -1


# ---------------------------------------------------------------
# LOCALITY EXTRACTION (regex)
# ---------------------------------------------------------------
# Localities follow patterns (sector 17, block a, anna nagar).
# Regex catches the pattern without needing a complete list.

LOCALITY_PATTERNS = [
    r"(sector\s*\d+[a-z]?)",       # sector 17, sector 4b
    r"(block\s*[a-z0-9]+)",        # block a, block 12
    r"(phase\s*[0-9ivx]+)",        # phase 1, phase iv
    r"(pocket\s*[a-z0-9]+)",       # pocket b
    r"(ward\s*\d+)",               # ward 5
    r"(\w+\s+nagar)",              # anna nagar, rohini nagar
    r"(\w+\s+colony)",             # shastri colony
    r"(\w+\s+enclave)",            # vasant enclave
    r"(\w+\s+vihar)",              # mayur vihar
    r"(\w+\s+layout)",             # yelahanka layout
]

def extract_locality(text):
    """Extract locality using regex patterns. Returns string or None."""
    for pattern in LOCALITY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().title()
    return None


# ---------------------------------------------------------------
# STATE EXTRACTION
# ---------------------------------------------------------------
# Also saves a city_hint when the matched state token is also a
# city alias. Example: "delhi" → State=Delhi, hint=New Delhi.
# City extraction uses this hint even after "delhi" is removed.

_city_hint = None

def extract_state(text):
    global _city_hint
    _city_hint = None

    states = _get_state_list()
    tokens = text.split()

    state, start, end = sliding_window_match(tokens, states, max_window=4, threshold=78, single_word_threshold=78)

    if state:
        # Check if matched token is also a city alias
        matched_tokens = " ".join(tokens[start:end])
        hint = alias_lookup(matched_tokens)
        if hint and hint in _get_city_list():
            _city_hint = hint

        remaining_tokens = tokens[:start] + tokens[end:]
        remaining = " ".join(remaining_tokens).strip()
        return state.title(), remaining

    return None, text


# ---------------------------------------------------------------
# CITY EXTRACTION
# ---------------------------------------------------------------

def extract_city(text):
    global _city_hint

    cities = _get_city_list()

    # Use hint from state extraction if available
    if _city_hint and _city_hint in cities:
        hint = _city_hint
        _city_hint = None
        return hint.title(), text

    tokens = text.split()
    city, start, end = sliding_window_match(tokens, cities, max_window=3, threshold=78, single_word_threshold=85)

    if city:
        remaining_tokens = tokens[:start] + tokens[end:]
        remaining = " ".join(remaining_tokens).strip()
        return city.title(), remaining

    return None, text


# ---------------------------------------------------------------
# MAIN PARSE FUNCTION
# ---------------------------------------------------------------
# Order: state → city → locality
# Each step removes matched tokens before passing text forward.

def parse_address(normalized_text, remaining_text=None):
    """
    Extract city, state, locality from a cleaned address string.

    Args:
        normalized_text : full cleaned address
        remaining_text  : text after pincode + house_no removed (preferred)

    Returns:
        dict with keys: city, state, locality
    """
    text = remaining_text if remaining_text else normalized_text

    state,    text_after_state = extract_state(text)
    city,     text_after_city  = extract_city(text_after_state)
    locality                   = extract_locality(text_after_city)

    return {
        "city":     city,
        "state":    state,
        "locality": locality,
    }


# ---------------------------------------------------------------
# TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("flat no 12 sector 17 dwarka delhi",            "City=New Delhi, State=Delhi, Locality=Sector 17"),
        ("h 34 kormangla 5th block bangalore",            "City=Bengaluru Urban, Locality=Block..."),
        ("plot 9 near metro station noida uttar pradesh", "City=Gautam Buddha Nagar, State=Uttar Pradesh"),
        ("12 b andheri west mumbai",                      "City=Mumbai"),
        ("sector 62 gurgaon haryana",                     "City=Gurugram, State=Haryana, Locality=Sector 62"),
        ("42 anna nagar chennai tamil nadu",              "City=Chennai, State=Tamil Nadu, Locality=Anna Nagar"),
        ("dlehi rohini",                                  "State=Delhi (typo)"),
        ("bangalor koramangla",                           "City=Bengaluru Urban (typo via alias)"),
    ]

    print("=" * 60)
    for text, expected in test_cases:
        result = parse_address(text)
        print(f"Input    : {text}")
        print(f"Expected : {expected}")
        print(f"City     : {result['city']}")
        print(f"State    : {result['state']}")
        print(f"Locality : {result['locality']}")
        print("-" * 60)