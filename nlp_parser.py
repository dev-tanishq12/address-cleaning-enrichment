"""
nlp_parser.py
-------------
Extracts city, state, and locality from a cleaned Indian address string.
- Fuzzy and Exact String Matching for 700+ Indian districts & 36 states/UTs
- Non-greedy token matching to prevent swallowing adjacent numbers or locality names
- Comprehensive locality parsing (patterns + semantic fallback)
- Cross-geographic resolution (City -> State, State -> City for UTs like Delhi/Chandigarh)
"""

import re
try:
    from rapidfuzz import process as fuzz_process, fuzz
    RAPIDFUZZ = True
except ImportError:
    import difflib
    RAPIDFUZZ = False

from pincode_lookup import get_all_cities, get_all_states, get_state_for_district


# ---------------------------------------------------------------
# CITY & DISTRICT ALIAS MAP
# ---------------------------------------------------------------
CITY_ALIASES = {
    # Delhi NCR
    "delhi":          "new delhi",
    "dilli":          "new delhi",
    "new delhi":      "new delhi",
    "gurgaon":        "gurugram",
    "gurugram":       "gurugram",
    "noida":          "gautam buddha nagar",
    "greater noida":  "gautam buddha nagar",
    "ghaziabad":      "ghaziabad",
    "faridabad":      "faridabad",

    # Karnataka
    "bangalore":      "bengaluru urban",
    "bangaluru":      "bengaluru urban",
    "bengaluru":      "bengaluru urban",
    "mysore":         "mysuru",
    "mangalore":      "dakshina kannada",
    "belgaum":        "belagavi",
    "hubli":          "dharwad",

    # Maharashtra
    "bombay":         "mumbai",
    "mumbai":         "mumbai",
    "navi mumbai":    "thane",
    "thane":          "thane",
    "pune":           "pune",
    "poona":          "pune",
    "nagpur":         "nagpur",
    "nashik":         "nashik",
    "nasik":          "nashik",
    "aurangabad":     "aurangabad",

    # Tamil Nadu
    "madras":         "chennai",
    "chennai":        "chennai",
    "coimbatore":     "coimbatore",
    "madurai":        "madurai",
    "trichy":         "tiruchirappalli",
    "salem":          "salem",

    # West Bengal
    "calcutta":       "kolkata",
    "kolkata":        "kolkata",
    "howrah":         "howrah",
    "asansol":        "paschim bardhaman",

    # Telangana & AP
    "hyderabad":      "hyderabad",
    "secunderabad":   "hyderabad",
    "vizag":          "visakhapatnam",
    "visakhapatnam":  "visakhapatnam",
    "vijayawada":     "krishna",

    # Gujarat
    "ahmedabad":      "ahmedabad",
    "surat":          "surat",
    "baroda":         "vadodara",
    "vadodara":       "vadodara",
    "rajkot":         "rajkot",

    # Kerala
    "cochin":         "ernakulam",
    "kochi":          "ernakulam",
    "trivandrum":     "thiruvananthapuram",
    "calicut":        "kozhikode",

    # North & Central
    "jaipur":         "jaipur",
    "lucknow":        "lucknow",
    "kanpur":         "kanpur nagar",
    "varanasi":       "varanasi",
    "banaras":        "varanasi",
    "patna":          "patna",
    "bhopal":         "bhopal",
    "indore":         "indore",
    "chandigarh":     "chandigarh",
    "dehradun":       "dehradun",
    "shimla":         "shimla",
    "amritsar":       "amritsar",
    "ludhiana":       "ludhiana",
    "jalandhar":      "jalandhar",
    "ranchi":         "ranchi",
    "bhubaneswar":    "khordha",
    "cuttack":        "cuttack",
    "guwahati":       "kamrup metropolitan",
}

_CITY_LIST       = None
_CITY_SET        = None
_STATE_LIST      = None
_STATE_SET       = None

def _get_city_list():
    global _CITY_LIST, _CITY_SET
    if _CITY_LIST is None:
        _CITY_LIST = [c.lower() for c in get_all_cities()]
        _CITY_SET  = set(_CITY_LIST)
    return _CITY_LIST

def _get_state_list():
    global _STATE_LIST, _STATE_SET
    if _STATE_LIST is None:
        _STATE_LIST = [s.lower() for s in get_all_states()]
        _STATE_SET  = set(_STATE_LIST)
    return _STATE_LIST


# ---------------------------------------------------------------
# PRECISE SLIDING WINDOW MATCHING
# ---------------------------------------------------------------
def smart_sliding_match(tokens, choices, alias_map=None, threshold=82):
    if not tokens:
        return None, -1, -1

    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)
    choices_set = set(choices)

    # 1. Exact alias matches (e.g. "bangalore" -> "bengaluru urban")
    if alias_map:
        for w in [3, 2, 1]:
            for i in range(n - w + 1):
                candidate = " ".join(tokens_lower[i : i + w])
                if candidate in alias_map:
                    official = alias_map[candidate]
                    if official in choices_set:
                        return official, i, i + w

    # 2. Exact choice match (e.g. "mumbai", "new delhi")
    for w in [3, 2, 1]:
        for i in range(n - w + 1):
            candidate = " ".join(tokens_lower[i : i + w])
            if candidate in choices_set:
                return candidate, i, i + w

    # 3. Fuzzy match against aliases (e.g. "bangalor")
    if alias_map:
        alias_keys = list(alias_map.keys())
        for w in [1, 2]:
            for i in range(n - w + 1):
                candidate = " ".join(tokens_lower[i : i + w])
                if len(candidate) < 5:
                    continue
                if RAPIDFUZZ:
                    res = fuzz_process.extractOne(candidate, alias_keys, scorer=fuzz.ratio, score_cutoff=86)
                    if res:
                        matched_key = res[0]
                        official = alias_map[matched_key]
                        if official in choices_set:
                            return official, i, i + w

    # 4. Fuzzy match against official choices (word-count aligned)
    best_match = None
    best_score = 0
    best_pos   = (-1, -1)

    for w in [1, 2, 3]:
        for i in range(n - w + 1):
            candidate = " ".join(tokens_lower[i : i + w])
            if len(candidate) < 4:
                continue

            choices_subset = [c for c in choices if len(c.split()) == w]
            if not choices_subset:
                continue

            if RAPIDFUZZ:
                res = fuzz_process.extractOne(candidate, choices_subset, scorer=fuzz.ratio, score_cutoff=threshold)
                if res:
                    matched_choice, score, _ = res
                    if score > best_score:
                        best_match = matched_choice
                        best_score = score
                        best_pos   = (i, i + w)
            else:
                for c in choices_subset:
                    score = difflib.SequenceMatcher(None, candidate, c).ratio() * 100
                    if score >= threshold and score > best_score:
                        best_match = c
                        best_score = score
                        best_pos   = (i, i + w)

    if best_match:
        return best_match, best_pos[0], best_pos[1]

    return None, -1, -1


# ---------------------------------------------------------------
# STATE & CITY EXTRACTION
# ---------------------------------------------------------------
def extract_state(tokens):
    states = _get_state_list()
    state, start, end = smart_sliding_match(tokens, states, alias_map=None, threshold=80)
    if state:
        rem_tokens = tokens[:start] + tokens[end:]
        return state.title(), rem_tokens
    return None, tokens

def extract_city(tokens):
    cities = _get_city_list()
    city, start, end = smart_sliding_match(tokens, cities, alias_map=CITY_ALIASES, threshold=82)
    if city:
        rem_tokens = tokens[:start] + tokens[end:]
        return city.title(), rem_tokens
    return None, tokens


# ---------------------------------------------------------------
# LOCALITY EXTRACTION
# ---------------------------------------------------------------
LOCALITY_PATTERNS = [
    r"\b(sector\s*[-/]?\s*\d+[a-z]?)\b",
    r"\b(phase\s*[-/]?\s*[0-9ivx]+)\b",
    r"\b(block\s*[-/]?\s*[a-z0-9]+)\b",
    r"\b(pocket\s*[-/]?\s*[a-z0-9]+)\b",
    r"\b(ward\s*[-/]?\s*\d+)\b",
    r"\b([a-z0-9]+\s+nagar)\b",
    r"\b([a-z0-9]+\s+colony)\b",
    r"\b([a-z0-9]+\s+enclave)\b",
    r"\b([a-z0-9]+\s+vihar)\b",
    r"\b([a-z0-9]+\s+layout)\b",
    r"\b([a-z0-9]+\s+road)\b",
    r"\b([a-z0-9]+\s+marg)\b",
    r"\b([a-z0-9]+\s+street)\b",
    r"\b([a-z0-9]+\s+lane)\b",
    r"\b([a-z0-9]+\s+market)\b",
    r"\b([a-z0-9]+\s+bazaar)\b",
    r"\b([a-z0-9]+\s+heights)\b",
    r"\b([a-z0-9]+\s+towers)\b",
    r"\b([a-z0-9]+\s+residency)\b",
    r"\b([a-z0-9]+\s+apartments?)\b",
]

def extract_locality(tokens):
    text = " ".join(tokens)
    
    # 1. Regex patterns
    for pat in LOCALITY_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()

    # 2. Semantic fallback on remaining tokens
    stopwords = {
        "near", "opp", "opposite", "behind", "adjacent", "adj", "nr", "at",
        "floor", "flr", "nd", "th", "rd", "st", "village", "post", "dist",
        "station", "metro", "-", "/", "#", "flat", "plot", "house", "room",
        "water", "tank", "point", "gate", "stop"
    }
    clean_words = [w for w in tokens if w.lower() not in stopwords and not w.isdigit() and len(w) > 1]
    
    if clean_words:
        candidate = " ".join(clean_words).strip().title()
        if len(candidate) >= 3:
            return candidate

    return None


# ---------------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------------
def parse_address(normalized_text, remaining_text=None):
    text = remaining_text if remaining_text else normalized_text
    tokens = text.split()

    # 1. Extract State
    state, tokens_after_state = extract_state(tokens)

    # 2. Extract City
    city, tokens_after_city = extract_city(tokens_after_state)

    # 3. Dual-role State/City Resolution (Delhi, Chandigarh, Goa)
    if state and not city:
        if state.lower() in ("delhi", "new delhi"):
            city = "New Delhi"
        elif state.lower() == "chandigarh":
            city = "Chandigarh"

    # 4. Infer State from City if State is missing
    if city and not state:
        inferred_state = get_state_for_district(city)
        if inferred_state:
            state = inferred_state

    # 5. Extract Locality
    locality = extract_locality(tokens_after_city)

    return {
        "city":     city,
        "state":    state,
        "locality": locality,
    }


if __name__ == "__main__":
    test_cases = [
        "flat no 12 sector 17 dwarka delhi",
        "koramangala 5th block bangalore",
        "near metro station noida uttar pradesh",
        "andheri west mumbai",
        "sector 62 gurgaon haryana",
        "sector 9 ambala haryana",
        "anna nagar chennai tamil nadu",
    ]
    for tc in test_cases:
        res = parse_address(tc)
        print(f"Text     : {tc}")
        print(f"City     : {res['city']}")
        print(f"State    : {res['state']}")
        print(f"Locality : {res['locality']}")
        print("-" * 50)
