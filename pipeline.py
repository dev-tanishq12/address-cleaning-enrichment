"""
pipeline.py
-----------
End-to-End Address Cleaning & Postal Enrichment Pipeline.

Flow:
    raw address string
        → rule_engine.py      : normalize text, expand abbreviations, extract house_no & pincode
        → nlp_parser.py       : extract city, state, locality with cross-geographic resolution
        → pincode_lookup.py   : validate pincode OR reverse-infer pincode/district/coords if missing
        → confidence_score.py : balanced ML confidence scoring & categorization
        → structured enrichment output
"""

import argparse
from rule_engine          import rule_based_cleaning
from pincode_lookup       import lookup_pincode, reverse_lookup, get_state_for_district
from nlp_parser           import parse_address
from confidence_score     import compute_confidence
from discrepancy_detector import validate_address_discrepancy
from deliverability_index import assess_deliverability


# ---------------------------------------------------------------
# PROCESS ADDRESS
# ---------------------------------------------------------------
def process_address(raw_address):
    """
    Run the complete cleaning, enrichment, and validation pipeline
    on a raw address string.
    """
    if not raw_address or not isinstance(raw_address, str):
        return {"error": "Invalid address input"}

    # ── Step 1: Rule-Based Cleaning & Syntactic Extraction ────────
    rule_res   = rule_based_cleaning(raw_address)
    normalized = rule_res["normalized_address"]
    house_no   = rule_res["house_no"]
    pincode    = rule_res["pincode"]
    landmark   = rule_res["landmark"]
    phone      = rule_res["phone_number"]
    remaining  = rule_res["remaining_text"]

    # ── Step 2: NLP Parsing ───────────────────────────────────────
    nlp_res  = parse_address(normalized, remaining)
    city     = nlp_res["city"]
    state    = nlp_res["state"]
    locality = nlp_res["locality"]

    # ── Step 3: Postal Lookup / Reverse Inference ─────────────────
    district          = None
    latitude          = None
    longitude         = None
    pincode_source    = None
    pincode_validated = False
    pincode_error     = None
    enrichment_notes  = []
    pin_data          = None

    if pincode:
        # Pincode was directly provided in the text
        pin_data = lookup_pincode(pincode)
        if pin_data:
            district          = pin_data["district"]
            state             = state or pin_data["state"]
            latitude          = pin_data["latitude"]
            longitude         = pin_data["longitude"]
            pincode_source    = "provided"
            pincode_validated = True
            enrichment_notes.append(f"Pincode {pincode} verified via India Post DB")
        else:
            pincode_source    = "provided"
            pincode_validated = False
            pincode_error     = "Pincode not found in official postal dataset"
            enrichment_notes.append("Pincode could not be validated against postal database")
    else:
        # No pincode provided: run Reverse Lookup
        rev = reverse_lookup(locality=locality, city=city, state=state, full_text=normalized)
        if rev:
            if rev.get("pincode"):
                pincode           = rev["pincode"]
                pincode_source    = "inferred"
                pincode_validated = True
                office = rev.get("matched_office")
                enrichment_notes.append(f"Pincode {pincode} inferred from {office or locality}")

            district  = rev.get("district") or district
            state     = state or rev.get("state")
            latitude  = rev.get("latitude")
            longitude = rev.get("longitude")

            if rev.get("source") == "district_fallback":
                enrichment_notes.append(f"Coordinates estimated from {city or district} district center")
        else:
            enrichment_notes.append("Pincode not provided; locality too ambiguous to infer postal code")

    # Final geographic sanity checks
    if city and not state:
        state = get_state_for_district(city)
    if city and not district:
        district = city

    # ── Step 4: Cross-Validation & Discrepancy Detection ──────────
    discrepancy = validate_address_discrepancy(
        user_state=state,
        user_city=city,
        pincode=pincode if pincode_source == "provided" else None,
        pin_data=pin_data
    )
    if discrepancy.get("has_conflict"):
        enrichment_notes.append(f"Alert: {discrepancy['message']}")

    record = {
        "raw_address":        raw_address,
        "normalized_address": normalized,
        "house_no":           house_no,
        "locality":           locality,
        "landmark":           landmark,
        "phone_number":       phone,
        "city":               city,
        "district":           district,
        "state":              state,
        "pincode":            pincode,
        "pincode_source":     pincode_source,
        "pincode_validated":  pincode_validated,
        "pincode_error":      pincode_error,
        "discrepancy":        discrepancy,
        "latitude":           latitude,
        "longitude":          longitude,
        "enrichment_notes":   enrichment_notes,
    }

    # ── Step 5: Confidence Scoring ────────────────────────────────
    conf = compute_confidence(record)
    record["confidence_score"] = conf["score"]
    record["confidence_tier"]  = conf["tier"]
    record["scoring_method"]   = conf["method"]

    # ── Step 6: Logistics Deliverability & RTO Index ──────────────
    record["deliverability"] = assess_deliverability(record)

    return record


# ---------------------------------------------------------------
# PRETTY PRINT
# ---------------------------------------------------------------
def print_result(res):
    print(f"  raw_address        : {res['raw_address']}")
    print(f"  normalized_address : {res['normalized_address']}")
    print(f"  house_no           : {res['house_no']}")
    print(f"  locality           : {res['locality']}")
    print(f"  landmark           : {res.get('landmark')}")
    print(f"  phone_number       : {res.get('phone_number')}")
    print(f"  city               : {res['city']}")
    print(f"  district           : {res['district']}")
    print(f"  state              : {res['state']}")
    print(f"  pincode            : {res['pincode']} ({res.get('pincode_source', 'none')})")
    print(f"  coordinates        : ({res['latitude']}, {res['longitude']})")
    print(f"  confidence_score   : {res['confidence_score']}  ({res['confidence_tier']})")
    print(f"  scoring_method     : {res['scoring_method']}")
    if res.get("deliverability"):
        deliv = res["deliverability"]
        print(f"  deliverability     : {deliv['deliverability_tier']} (RTO Risk: {deliv['rto_risk']} - {deliv['estimated_success_rate']}% success)")
        print(f"  dispatch_action    : {deliv['dispatch_action']}")
        print(f"  delivery_checks    : {deliv['checklist_passed']}/{deliv['checklist_total']} passed")
    if res.get("enrichment_notes"):
        print(f"  notes              : {'; '.join(res['enrichment_notes'])}")


# ---------------------------------------------------------------
# SAMPLE TEST SUITE
# ---------------------------------------------------------------
SAMPLE_ADDRESSES = [
    "Flat no 12, Sec 17 Dwarka, Delhi - 110078",   # complete with pincode
    "H-34 kormangla 5th blk blr",                  # NO pincode -> should infer pincode + score HIGH/MEDIUM
    "Plot 9, near metro stn, noida up 201301",      # complete with pincode
    "12/B Andheri w mmbai",                         # NO pincode -> should infer pincode + score HIGH/MEDIUM
    "sector-62 gurgaon haryana",                    # NO pincode, no house_no -> should score MEDIUM
    "near water tank village up",                   # vague -> should score LOW
    "sector 9 ambala haryana 134003",              # complete with pincode
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Address Cleaning Pipeline")
    parser.add_argument("--address", type=str, help="Single address to process")
    args = parser.parse_args()

    if args.address:
        print("\nProcessing address...")
        print("-" * 55)
        res = process_address(args.address)
        print_result(res)
    else:
        print("=" * 60)
        print("  ENHANCED ADDRESS CLEANING & ENRICHMENT PIPELINE")
        print("=" * 60)
        for addr in SAMPLE_ADDRESSES:
            print(f"\nInput: {addr}")
            print("-" * 60)
            res = process_address(addr)
            print_result(res)
            print()
