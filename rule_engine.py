"""
rule_engine.py
--------------
Rule-based text cleaning and normalization for Indian addresses.
Features:
- Indian mobile/contact phone number detection & extraction
- Landmark & Point of Interest (POI) extraction (near, opp, behind, next to)
- Text normalization & abbreviations expansion
- Structured house/flat/plot number extraction
- 6-digit Pincode extraction (handles spaces and hyphens)
"""

import re

# ---------------------------------------------------------------
# ABBREVIATION DICTIONARY
# ---------------------------------------------------------------
ABBREVIATIONS = {
    # Cities & Metros
    "blr":       "bangalore",
    "bnglr":     "bangalore",
    "bengaluru": "bangalore",
    "mmbai":     "mumbai",
    "bom":       "mumbai",
    "hyd":       "hyderabad",
    "del":       "delhi",
    "dilli":     "delhi",
    "ggn":       "gurgaon",
    "gurugram":  "gurgaon",
    "fbd":       "faridabad",
    "chd":       "chandigarh",
    "kol":       "kolkata",
    "cal":       "kolkata",
    "maa":       "chennai",
    "pnq":       "pune",
    "amd":       "ahmedabad",

    # States
    "up":        "uttar pradesh",
    "mp":        "madhya pradesh",
    "ap":        "andhra pradesh",
    "hp":        "himachal pradesh",
    "tn":        "tamil nadu",
    "ka":        "karnataka",
    "ktk":       "karnataka",
    "mh":        "maharashtra",
    "rj":        "rajasthan",
    "pb":        "punjab",
    "uk":        "uttarakhand",
    "wb":        "west bengal",
    "kl":        "kerala",

    # Urban Planning & Localities
    "sec":       "sector",
    "sect":      "sector",
    "blk":       "block",
    "ph":        "phase",
    "pkt":       "pocket",
    "wrd":       "ward",
    "ngr":       "nagar",
    "col":       "colony",
    "enc":       "enclave",
    "ext":       "extension",
    "extn":      "extension",
    "soc":       "society",
    "apt":       "apartment",
    "apts":      "apartments",
    "bldg":      "building",
    "flt":       "flat",
    "vill":      "village",
    "po":        "post office",
    "dist":      "district",

    # Roads & Directions
    "rd":        "road",
    "st":        "street",
    "stn":       "station",
    "ave":       "avenue",
    "hwy":       "highway",
    "w":         "west",
    "e":         "east",
    "n":         "north",
    "s":         "south",
    "nr":        "near",
    "opp":       "opposite",
    "adj":       "adjacent",
    "bhnd":      "behind",

    # Common locality typos
    "kormangla": "koramangala",
    "koramangla":"koramangala",
    "indranagar":"indiranagar",
}


# ---------------------------------------------------------------
# PHONE NUMBER EXTRACTION (Delivery Contact)
# ---------------------------------------------------------------
def extract_phone_number(raw_text):
    """
    Extract Indian mobile numbers (starts with 6-9, 10 digits, optional +91/0 prefix).
    Strips the phone number and its label (call, mob, phone, contact) from the text
    so it does not corrupt house number or pincode extraction.
    """
    if not raw_text:
        return None, raw_text

    trigger_pattern = r'\b(?:call|phone|ph|mobile|mob|contact|tel)\s*(?:no\.?|#)?:?\s*(?:\+?91|0)?[ -]?[6-9]\d{4}[ -]?\d{5}\b'
    plain_pattern   = r'(?:\+?91|0)?[ -]?[6-9]\d{4}[ -]?\d{5}\b'

    m = re.search(trigger_pattern, raw_text, re.IGNORECASE)
    if not m:
        m = re.search(plain_pattern, raw_text)

    if m:
        match_str = m.group()
        phone_match = re.search(plain_pattern, match_str)
        if phone_match:
            digits = re.sub(r'\D', '', phone_match.group())
            if len(digits) == 10:
                formatted = f"+91-{digits}"
            elif len(digits) == 11 and digits.startswith('0'):
                formatted = f"+91-{digits[1:]}"
            elif len(digits) == 12 and digits.startswith('91'):
                formatted = f"+91-{digits[2:]}"
            else:
                formatted = phone_match.group()

            cleaned = raw_text[:m.start()] + " " + raw_text[m.end():]
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return formatted, cleaned

    return None, raw_text


# ---------------------------------------------------------------
# LANDMARK / POI EXTRACTION
# ---------------------------------------------------------------
def extract_landmark(text):
    """
    Extract Indian landmark references (e.g. 'near metro station', 'opp shiva temple',
    'behind reliance fresh', 'next to hp petrol pump').
    Strips the landmark clause so it doesn't pollute the locality field.
    """
    if not text:
        return None, text

    pattern = r'\b(?:near|opposite to|opposite|opp\.?|behind|bhnd|next to|beside|adjacent to|adj\.?|in front of)\s+([a-z0-9\s/-]+?)(?=(?:,\s*|\s+(?:sector|block|delhi|mumbai|bangalore|kolkata|chennai|noida|gurgaon|hyderabad|\d{6})|$))'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        landmark_phrase = m.group().strip()
        landmark_clean = landmark_phrase.title()

        cleaned = text[:m.start()] + " " + text[m.end():]
        cleaned = re.sub(r'\s+', ' ', cleaned).strip(' ,')
        return landmark_clean, cleaned

    return None, text


# ---------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------
def normalize_text(address):
    """
    Standardize text casing, clean punctuation, preserve delimiters (/, -, #).
    """
    if not address or not isinstance(address, str):
        return ""

    address = address.lower()
    # Normalize spaced or hyphenated pincodes: "110 078" or "110-078" -> "110078"
    address = re.sub(r'\b(\d{3})\s+(\d{3})\b', r'\1\2', address)
    address = re.sub(r'\b(\d{3})-(\d{3})\b', r'\1\2', address)
    address = re.sub(r'[^a-z0-9\s/#-]', ' ', address)
    address = re.sub(r'\s+', ' ', address).strip()
    return address


# ---------------------------------------------------------------
# ABBREVIATION EXPANSION
# ---------------------------------------------------------------
def expand_abbreviations(address):
    words = address.split()
    expanded = [ABBREVIATIONS.get(word, word) for word in words]
    return " ".join(expanded)


# ---------------------------------------------------------------
# PINCODE EXTRACTION
# ---------------------------------------------------------------
def extract_pincode(address):
    match = re.search(r'\b[1-9]\d{5}\b', address)
    return match.group() if match else None


# ---------------------------------------------------------------
# HOUSE / UNIT NUMBER EXTRACTION
# ---------------------------------------------------------------
def extract_house_no(address):
    # Pattern 1: Property prefix + number (e.g., "flat no 12", "plot 9a", "house-45", "unit 3b")
    match = re.search(r'\b(?:flat|plot|house|villa|shop|room|office|h|unit)\s*(?:no\.?|number|#|-)?\s*[a-z0-9/-]+', address, re.IGNORECASE)
    if match:
        val = match.group().strip()
        if any(c.isdigit() for c in val):
            return val

    # Pattern 2: Explicit "no" or "#" + number
    match = re.search(r'\b(?:no\.?|#)\s*[a-z0-9/-]+', address, re.IGNORECASE)
    if match and any(c.isdigit() for c in match.group()):
        return match.group().strip()

    # Pattern 3: Fractional or slashed number (e.g. "12/b", "45/2")
    match = re.search(r'\b\d+/[a-z0-9-]+\b', address, re.IGNORECASE)
    if match:
        return match.group().strip()

    # Pattern 4: Alpha-numeric block/unit (e.g. "b-12", "a-402")
    match = re.search(r'\b[a-z]\s*[-/]?\s*\d+[a-z]?\b', address, re.IGNORECASE)
    if match:
        val = match.group().strip()
        if len(val) >= 2:
            return val

    return None


# ---------------------------------------------------------------
# MAIN CLEANING PIPELINE
# ---------------------------------------------------------------
def rule_based_cleaning(raw_address):
    # 1. First extract and strip phone numbers before digit parsing
    phone, text_no_phone = extract_phone_number(raw_address)

    # 2. Extract landmark and strip it
    landmark, text_no_lm = extract_landmark(text_no_phone)

    # 3. Normalize text and expand abbreviations
    normalized = normalize_text(text_no_lm)
    expanded   = expand_abbreviations(normalized)

    # 4. Extract pincode and house number
    pincode    = extract_pincode(expanded)
    house_no   = extract_house_no(expanded)

    # 5. Build clean remaining_text for NLP parsing
    remaining = expanded
    if pincode:
        remaining = re.sub(r'\b' + re.escape(pincode) + r'\b', ' ', remaining)
    if house_no:
        remaining = re.sub(r'\b' + re.escape(house_no) + r'\b', ' ', remaining)

    remaining = re.sub(r'\s+', ' ', remaining).strip(' /-')

    return {
        "raw_address":        raw_address,
        "normalized_address": expanded,
        "phone_number":       phone,
        "landmark":           landmark,
        "pincode":            pincode,
        "house_no":           house_no,
        "remaining_text":     remaining,
    }


if __name__ == "__main__":
    test_cases = [
        "Flat no 12, Sec 17 Dwarka, near metro station, Delhi - 110078 call 9876543210",
        "Plot 9, opp shiva temple, noida up 201301 ph: +91 9811223344",
        "H-34 kormangla 5th blk blr behind post office",
        "12/B Andheri w mmbai next to reliance fresh mob 08765432109",
        "sector-62 gurgaon haryana",
    ]
    for tc in test_cases:
        res = rule_based_cleaning(tc)
        print(f"Input    : {res['raw_address']}")
        print(f"Phone    : {res['phone_number']}")
        print(f"Landmark : {res['landmark']}")
        print(f"House No : {res['house_no']}")
        print(f"Pincode  : {res['pincode']}")
        print(f"Remaining: {res['remaining_text']}")
        print("-" * 60)
