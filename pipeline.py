"""
pipeline.py
-----------
Main pipeline — connects all four modules into one system.

Flow:
    raw address string
        → rule_engine.py     : clean text, extract pincode + house_no
        → pincode_lookup.py  : enrich with district, state, lat, lon
        → nlp_parser.py      : extract city, state, locality from text
        → confidence_score.py: score the result (ML or rule-based)
        → structured dict output

Usage:
    python pipeline.py                        # runs sample addresses
    python pipeline.py --address "your address here"  # single address
"""

import argparse
from rule_engine      import rule_based_cleaning
from pincode_lookup   import lookup_pincode
from nlp_parser       import parse_address
from confidence_score import compute_confidence


# ---------------------------------------------------------------
# CORE: PROCESS ONE ADDRESS
# ---------------------------------------------------------------

def process_address(raw_address):
    """
    Run the full pipeline on a single raw address string.

    Args:
        raw_address : str — the messy input address

    Returns:
        dict with all extracted and enriched fields + confidence score
    """

    # ── Step 1: Rule-based cleaning ──────────────────────────────
    # normalize text, expand abbreviations,
    # extract pincode and house number
    rule_result = rule_based_cleaning(raw_address)

    normalized  = rule_result["normalized_address"]
    house_no    = rule_result["house_no"]
    pincode     = rule_result["pincode"]
    remaining   = rule_result["remaining_text"] if "remaining_text" in rule_result else normalized

    # ── Step 2: Pincode lookup ────────────────────────────────────
    # use pincode to get district, state, lat, lon from India Post DB
    # returns empty dict with None values if pincode is None or not found
    if pincode:
        pin_data = lookup_pincode(pincode)
        if pin_data is None:
            pin_data = {"district": None, "state": None,
                        "latitude": None, "longitude": None}
    else:
        pin_data = {"district": None, "state": None,
                    "latitude": None, "longitude": None}

    # ── Step 3: NLP parsing ───────────────────────────────────────
    # extract city, state, locality from the cleaned text
    # pass remaining_text (pincode + house already removed) for cleaner input
    nlp_result = parse_address(normalized, remaining)

    # ── Step 4: Merge results ─────────────────────────────────────
    # PRIORITY RULE (derived from ML feature importance):
    #   pincode DB fields > NLP fields
    #   Because pincode_validated was 80% of model importance —
    #   database lookup is far more reliable than fuzzy text matching.
    #
    # Field sources:
    #   district  → DB only  (NLP doesn't extract district)
    #   state     → DB first, NLP as fallback
    #   city      → NLP only (DB doesn't have city-level data)
    #   locality  → NLP only (DB doesn't have locality data)
    #   lat/lon   → DB only  (NLP can't produce coordinates)

    district = pin_data["district"]
    state    = pin_data["state"] or nlp_result["state"]
    city     = nlp_result["city"]
    locality = nlp_result["locality"]
    latitude = pin_data["latitude"]
    longitude= pin_data["longitude"]

    # ── Step 5: Build record ──────────────────────────────────────
    record = {
        "raw_address":        raw_address,
        "normalized_address": normalized,
        "house_no":           house_no,
        "locality":           locality,
        "city":               city,
        "district":           district,
        "state":              state,
        "pincode":            pincode,
        "latitude":           latitude,
        "longitude":          longitude,
    }

    # ── Step 6: Confidence scoring ────────────────────────────────
    # tries ML model first, falls back to rule-based
    conf = compute_confidence(record)
    record["confidence_score"] = conf["score"]
    record["scoring_method"]   = conf["method"]

    # Confidence tier — easy to read label
    score = conf["score"]
    if score >= 70:
        record["confidence_tier"] = "HIGH"
    elif score >= 40:
        record["confidence_tier"] = "MEDIUM"
    else:
        record["confidence_tier"] = "LOW"

    return record


# ---------------------------------------------------------------
# PRETTY PRINT
# ---------------------------------------------------------------

def print_result(result):
    """Print pipeline output in a readable format."""
    print(f"  raw_address        : {result['raw_address']}")
    print(f"  normalized_address : {result['normalized_address']}")
    print(f"  house_no           : {result['house_no']}")
    print(f"  locality           : {result['locality']}")
    print(f"  city               : {result['city']}")
    print(f"  district           : {result['district']}")
    print(f"  state              : {result['state']}")
    print(f"  pincode            : {result['pincode']}")
    print(f"  latitude           : {result['latitude']}")
    print(f"  longitude          : {result['longitude']}")
    print(f"  confidence_score   : {result['confidence_score']}  ({result['confidence_tier']})")
    print(f"  scoring_method     : {result['scoring_method']}")


# ---------------------------------------------------------------
# SAMPLE ADDRESSES
# ---------------------------------------------------------------

SAMPLE_ADDRESSES = [
    "Flat no 12, Sec 17 Dwarka, Delhi - 110078",   # complete — should score HIGH
    "H-34 kormangla 5th blk blr",                  # no pincode — should score MEDIUM
    "Plot 9, near metro stn, noida up 201301",      # has pincode — should score HIGH
    "12/B Andheri w mmbai",                         # partial — should score LOW/MEDIUM
    "sector-62 gurgaon haryana",                    # no pincode — should score MEDIUM
    "near water tank village up",                   # very vague — should score LOW
    "sector 9 ambala haryana 134003",              # has pincode — should score HIGH
]


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Address Cleaning Pipeline")
    parser.add_argument("--address", type=str, help="Single address to process")
    args = parser.parse_args()

    if args.address:
        # Single address mode
        print("\nProcessing address...")
        print("-" * 55)
        result = process_address(args.address)
        print_result(result)
    else:
        # Demo mode — run all sample addresses
        print("=" * 55)
        print("  ADDRESS CLEANING & ENRICHMENT PIPELINE")
        print("=" * 55)

        for addr in SAMPLE_ADDRESSES:
            print(f"\nInput: {addr}")
            print("-" * 55)
            result = process_address(addr)
            print_result(result)
            print()