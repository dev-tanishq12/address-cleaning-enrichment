"""
pincode_lookup.py
-----------------
India Post Postal Database Lookup & Reverse Enrichment Engine.
Features:
- Instant O(1) dictionary lookup for Indian pincodes (19K+ pincodes, 165K+ post offices)
- High-performance reverse lookup by locality, city, and state
- District-to-State & City-to-State automatic resolution
- Geocoordinate enrichment (latitude, longitude as floats)
"""

import os
import re
import pandas as pd
from collections import defaultdict

try:
    from rapidfuzz import process as fuzz_process, fuzz
    RAPIDFUZZ = True
except ImportError:
    import difflib
    RAPIDFUZZ = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "india_post_pincode.csv")

# ---------------------------------------------------------------
# LOAD & PRE-INDEX POSTAL DATASET
# ---------------------------------------------------------------
# Read once into memory and construct fast O(1) hash maps.
df = pd.read_csv(DATA_PATH, dtype={"pincode": str})

# Clean basic string columns
df["pincode"] = df["pincode"].fillna("").astype(str).str.strip()
df["district"] = df["district"].fillna("").astype(str).str.strip().str.title()
df["statename"] = df["statename"].fillna("").astype(str).str.strip().str.title()
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Clean office names: strip post office classification suffixes (B.O, S.O, H.O, etc.)
df["office_clean"] = (
    df["officename"].fillna("").astype(str)
    .str.replace(r"\s+(?:B\.O|S\.O|H\.O|BO|SO|HO)\b", "", regex=True, flags=re.IGNORECASE)
    .str.strip()
    .str.title()
)

# 1. PINCODE DIRECT LOOKUP DICTIONARY: pincode -> dict
PINCODE_MAP = {}
for row in df[["pincode", "district", "statename", "latitude", "longitude"]].itertuples(index=False):
    pin = row[0]
    if pin and pin not in PINCODE_MAP:
        PINCODE_MAP[pin] = {
            "pincode":   pin,
            "district":  row[1] if row[1] else None,
            "state":     row[2] if row[2] else None,
            "latitude":  float(row[3]) if pd.notna(row[3]) else None,
            "longitude": float(row[4]) if pd.notna(row[4]) else None,
        }

# 2. DISTRICT / CITY TO STATE & COORDINATES
DISTRICT_TO_STATE = {}
DISTRICT_COORDS = {}
for row in df[["district", "statename", "latitude", "longitude"]].itertuples(index=False):
    d_clean = row[0].strip().title()
    s_clean = row[1].strip().title()
    if d_clean and d_clean not in DISTRICT_TO_STATE:
        DISTRICT_TO_STATE[d_clean.lower()] = s_clean
        if pd.notna(row[2]) and pd.notna(row[3]):
            DISTRICT_COORDS[d_clean.lower()] = (float(row[2]), float(row[3]))

# 3. INVERTED POST OFFICE INDEX FOR REVERSE LOOKUP
# Maps word token -> list of candidate post office dicts
OFFICE_WORD_INDEX = defaultdict(list)
# Maps state_lower -> set of office names (for fuzzy matching)
STATE_OFFICES = defaultdict(set)
# Maps district_lower -> set of office names
DISTRICT_OFFICES = defaultdict(set)

for row in df[["office_clean", "pincode", "district", "statename", "latitude", "longitude"]].itertuples(index=False):
    office = row[0]
    pin    = row[1]
    dist   = row[2] if row[2] else None
    state  = row[3] if row[3] else None
    lat    = float(row[4]) if pd.notna(row[4]) else None
    lon    = float(row[5]) if pd.notna(row[5]) else None

    if not office:
        continue

    rec = {
        "office":    office,
        "pincode":   pin,
        "district":  dist,
        "state":     state,
        "latitude":  lat,
        "longitude": lon,
    }

    if state:
        STATE_OFFICES[state.lower()].add(office)
    if dist:
        DISTRICT_OFFICES[dist.lower()].add(office)

    # Index distinctive alphanumeric tokens of length >= 3
    words = set(re.findall(r"[a-z0-9]+", office.lower()))
    for w in words:
        if len(w) >= 3:
            OFFICE_WORD_INDEX[w].append(rec)


# ---------------------------------------------------------------
# PINCODE LOOKUP (FORWARD)
# ---------------------------------------------------------------
def lookup_pincode(pincode):
    """
    Look up postal enrichment data for a given 6-digit Indian PIN code.
    Runs in O(1) constant time.
    """
    if pincode is None or not isinstance(pincode, str):
        return None

    pincode = pincode.strip()
    if not pincode.isdigit() or len(pincode) != 6:
        return None

    return PINCODE_MAP.get(pincode)


# ---------------------------------------------------------------
# REVERSE PINCODE & GEOGRAPHIC LOOKUP
# ---------------------------------------------------------------
def reverse_lookup(locality=None, city=None, state=None, full_text=None):
    """
    Infer pincode, district, state, and coordinates when no pincode is provided.
    Searches the 165K post office records filtered by state and/or district.

    Returns dict with inferred attributes, or None if no confident match found.
    """
    # Guard: if neither locality nor city is available, cannot reliably infer postal office
    if not locality and not city:
        return None

    # Normalize inputs
    target_state = state.lower().strip() if state else None
    target_dist  = city.lower().strip() if city else None

    # Derive state from district if state is missing
    if not target_state and target_dist and target_dist in DISTRICT_TO_STATE:
        target_state = DISTRICT_TO_STATE[target_dist].lower()

    # If locality is provided, search for specific post offices
    if locality:
        # Collect search keywords from locality only (to avoid pollution from state/city tokens)
        raw_tokens = re.findall(r"[a-z0-9]+", locality.lower())

        # Common generic words that should not trigger standalone matches
        stopwords = {
            "flat", "house", "plot", "near", "opposite", "opp", "road", "street",
            "delhi", "mumbai", "india", "floor", "flr", "block", "sector", "lane",
            "behind", "adjacent", "room", "unit", "building", "apartments",
            "village", "vill", "water", "tank", "chowk", "gate", "stop", "corner",
            "market", "bazaar", "bazar", "nagar", "colony", "enclave", "vihar",
            "area", "side", "cross", "main", "hall", "center", "centre"
        }
        query_tokens = [t for t in raw_tokens if len(t) >= 4 and t not in stopwords]
        num_tokens = [t for t in raw_tokens if t.isdigit() and len(t) <= 3]

        matched_recs = []

        # Strategy 1: Search inverted word index for specific locality tokens
        for token in query_tokens:
            hits = OFFICE_WORD_INDEX.get(token, [])
            for hit in hits:
                hit_dist = hit["district"].lower() if hit["district"] else ""
                hit_state = hit["state"].lower() if hit["state"] else ""

                # Score relevance
                score = 70
                if target_dist and (target_dist in hit_dist or hit_dist in target_dist):
                    score += 25
                elif target_state and target_state in hit_state:
                    score += 15
                else:
                    if target_dist or target_state:
                        continue

                # Bonus if numeric token also matches office name (e.g. Sec 17)
                office_lower = hit["office"].lower()
                for num in num_tokens:
                    if re.search(r'\b' + re.escape(num) + r'\b', office_lower):
                        score += 10

                matched_recs.append((hit, min(100, score)))

        # Strategy 2: Fuzzy match locality name against district/state offices
        if not matched_recs:
            search_choices = []
            if target_dist and target_dist in DISTRICT_OFFICES:
                search_choices = list(DISTRICT_OFFICES[target_dist])
            elif target_state and target_state in STATE_OFFICES:
                search_choices = list(STATE_OFFICES[target_state])

            if search_choices and RAPIDFUZZ:
                best = fuzz_process.extractOne(locality, search_choices, scorer=fuzz.token_sort_ratio, score_cutoff=75)
                if best:
                    matched_name, fuzz_score, _ = best
                    for token in re.findall(r"[a-z0-9]+", matched_name.lower()):
                        for hit in OFFICE_WORD_INDEX.get(token, []):
                            if hit["office"].lower() == matched_name.lower():
                                if target_state and hit["state"] and target_state not in hit["state"].lower():
                                    continue
                                matched_recs.append((hit, fuzz_score))
                                break
                        if matched_recs:
                            break

        if matched_recs:
            matched_recs.sort(key=lambda x: (x[1], 1 if x[0]["latitude"] is not None else 0), reverse=True)
            best_rec, conf_score = matched_recs[0]

            lat = best_rec["latitude"]
            lon = best_rec["longitude"]
            if lat is None and best_rec["district"] and best_rec["district"].lower() in DISTRICT_COORDS:
                lat, lon = DISTRICT_COORDS[best_rec["district"].lower()]

            return {
                "pincode":          best_rec["pincode"],
                "district":         best_rec["district"],
                "state":            best_rec["state"],
                "latitude":         lat,
                "longitude":        lon,
                "matched_office":   best_rec["office"],
                "match_confidence": conf_score,
                "source":           "inferred",
            }

    # Strategy 3: District-level fallback if district/city is known
    if target_dist and target_dist in DISTRICT_COORDS:
        lat, lon = DISTRICT_COORDS[target_dist]
        resolved_state = DISTRICT_TO_STATE.get(target_dist, state)
        return {
            "pincode":          None,
            "district":         target_dist.title(),
            "state":            resolved_state,
            "latitude":         lat,
            "longitude":        lon,
            "matched_office":   None,
            "match_confidence": 60,
            "source":           "district_fallback",
        }

    return None


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def get_all_cities():
    """Returns sorted unique district names."""
    return sorted(list({d for d in df["district"].dropna().unique() if d}))

def get_all_states():
    """Returns sorted unique state names."""
    return sorted(list({s for s in df["statename"].dropna().unique() if s}))

def get_state_for_district(district_name):
    """Resolve state for a known city/district."""
    if not district_name:
        return None
    return DISTRICT_TO_STATE.get(district_name.lower().strip())


if __name__ == "__main__":
    print("Testing forward lookup:")
    print("110078:", lookup_pincode("110078"))

    print("\nTesting reverse lookup:")
    print("Dwarka in Delhi ->", reverse_lookup(locality="Dwarka", city="New Delhi", state="Delhi"))
    print("Koramangala in Bangalore ->", reverse_lookup(locality="Koramangala 5th Block", city="Bengaluru Urban", state="Karnataka"))
    print("Andheri West in Mumbai ->", reverse_lookup(locality="Andheri West", city="Mumbai", state="Maharashtra"))
