import re

# ---------------------------------------------------------------
# TEST DATA
# ---------------------------------------------------------------
test_addresses = [
    "Flat no 12, Sec 17 Dwarka, Delhi - 110078",
    "H-34 kormangla 5th blk blr",
    "Plot 9, near metro stn, noida up 201301",
    "12/B Andheri w mmbai",
    "sector-62 gurgaon haryana"
]


# ---------------------------------------------------------------
# ABBREVIATION DICTIONARY
# ---------------------------------------------------------------
ABBREVIATIONS = {
    "blr":   "bangalore",
    "bnglr": "bangalore",
    "mmbai": "mumbai",
    "hyd":   "hyderabad",
    "sec":   "sector",
    "stn":   "station",
    "w":     "west",
    "e":     "east",
    "blk":   "block",
    "del":   "delhi",
    "up":    "uttar pradesh",
    "nr":    "near",
}


# ---------------------------------------------------------------
# STEP 1 — TEXT NORMALIZATION
# ---------------------------------------------------------------
def normalize_text(address):
    address = address.lower()
    address = re.sub(r'[^a-z0-9\s/]', ' ', address)  # keep /
    address = re.sub(r'\s+', ' ', address).strip()
    return address


# ---------------------------------------------------------------
# STEP 2 — ABBREVIATION EXPANSION
# ---------------------------------------------------------------
def expand_abbreviations(address):
    words = address.split()
    expanded = [ABBREVIATIONS.get(word, word) for word in words]
    return " ".join(expanded)


# ---------------------------------------------------------------
# STEP 3 — PINCODE EXTRACTION
# ---------------------------------------------------------------
def extract_pincode(address):
    match = re.search(r'\b\d{6}\b', address)
    return match.group() if match else None


# ---------------------------------------------------------------
# STEP 4 — HOUSE NUMBER EXTRACTION
# ---------------------------------------------------------------
def extract_house_no(address):
    # Pattern 1: keyword-based  e.g. "flat 12", "h 34", "plot 9"
    match = re.search(r'\b(flat|plot|house|h|no)\s*\d+[a-z]?\b', address)
    if match:
        return match.group()

    # Pattern 2: number/letter  e.g. "12/b", "45/2"
    match = re.search(r'\b\d+/[a-z0-9]+\b', address)
    if match:
        return match.group()

    # Pattern 3: letter+number  e.g. "b12", "a7"
    match = re.search(r'\b[a-z]\d+\b', address)
    if match:
        return match.group()

    return None


# ---------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------
def rule_based_cleaning(raw_address):
    normalized = normalize_text(raw_address)
    expanded   = expand_abbreviations(normalized)
    pincode    = extract_pincode(expanded)
    house_no   = extract_house_no(expanded)

    # Build remaining_text — remove pincode and house_no from expanded text
    # WHY: nlp_parser works better without these already-extracted tokens
    # "flat no 12 sector 17 dwarka delhi 110078"
    # → after removal: "sector 17 dwarka delhi"
    # → cleaner input for city/state/locality extraction
    remaining = expanded
    if pincode:
        remaining = remaining.replace(pincode, "")
    if house_no:
        remaining = remaining.replace(house_no, "")
    remaining = " ".join(remaining.split())   # collapse extra spaces

    return {
        "raw_address":        raw_address,
        "normalized_address": expanded,
        "pincode":            pincode,
        "house_no":           house_no,
        "remaining_text":     remaining,      # used by nlp_parser
    }


# ---------------------------------------------------------------
# RUN TEST
# ---------------------------------------------------------------
if __name__ == "__main__":
    for addr in test_addresses:
        result = rule_based_cleaning(addr)
        print(f"Input    : {result['raw_address']}")
        print(f"Cleaned  : {result['normalized_address']}")
        print(f"Pincode  : {result['pincode']}")
        print(f"House No : {result['house_no']}")
        print(f"Remaining: {result['remaining_text']}")
        print("-" * 60)