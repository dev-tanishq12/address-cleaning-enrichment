"""
deliverability_index.py
-----------------------
Logistics Deliverability & RTO (Return-to-Origin) Risk Index.
Evaluates address hygiene and doorstep readiness for e-commerce,
hyperlocal (Swiggy/Zomato), and 3PL courier logistics (Delhivery, Ekart, BlueDart).
"""

def assess_deliverability(record):
    """
    Evaluates deliverability readiness, RTO failure risk, and recommended
    dispatch action based on extracted address elements.

    Returns dict:
        rto_risk               (str)  : 'LOW', 'MEDIUM', 'HIGH'
        deliverability_tier    (str)  : 'LAST_MILE_READY', 'HUB_LEVEL_ONLY', 'CRITICAL_CONFLICT', etc.
        estimated_success_rate (int)  : 0 - 100 percentage
        dispatch_action        (str)  : Actionable dispatch recommendation
        checklist              (dict) : Itemized checklist of 6 delivery factors
        checklist_passed       (int)  : Number of passed checks
        checklist_total        (int)  : Total checks (6)
    """
    discrepancy  = record.get("discrepancy") or {}
    has_conflict = discrepancy.get("has_conflict", False)
    conflict_sev = discrepancy.get("severity", "")

    has_house    = bool(record.get("house_no"))
    has_locality = bool(record.get("locality"))
    has_landmark = bool(record.get("landmark"))
    has_phone    = bool(record.get("phone_number"))
    has_coords   = bool(record.get("latitude") and record.get("longitude"))

    pin          = record.get("pincode")
    pin_valid    = record.get("pincode_validated", False)

    # 1. Six-Factor Deliverability Checklist
    checklist = {
        "unit_specified":     {"passed": has_house,    "label": "House / Flat / Unit Number"},
        "locality_mapped":    {"passed": has_locality, "label": "Locality / Sub-Area"},
        "landmark_available": {"passed": has_landmark, "label": "Navigational Landmark / POI"},
        "pincode_verified":   {"passed": pin_valid,    "label": "Postal PIN Code Verified"},
        "contact_available":  {"passed": has_phone,    "label": "Delivery Phone Contact"},
        "geocoded":           {"passed": has_coords,   "label": "Geocoordinates Available"},
    }

    passed_count = sum(1 for item in checklist.values() if item["passed"])

    # 2. Risk & Deliverability Persona
    if has_conflict and conflict_sev == "CRITICAL":
        rto_risk = "HIGH"
        persona  = "CRITICAL_CONFLICT"
        recommendation = "Hold Dispatch: Postal PIN code and state contradict each other. High risk of cross-state misrouting."
        success_rate = 15
    elif not pin_valid and not pin:
        rto_risk = "HIGH"
        persona  = "UNROUTABLE"
        recommendation = "Hold Dispatch: Missing postal code and unmapped locality. Trigger customer automated address update."
        success_rate = 25
    elif has_house and (has_locality or has_landmark) and pin_valid:
        rto_risk = "LOW"
        persona  = "LAST_MILE_READY"
        recommendation = "Direct Dispatch: High probability of first-attempt doorstep delivery success."
        success_rate = 95 if has_phone else 90
    elif (has_locality or has_landmark) and pin_valid and not has_house:
        rto_risk = "MEDIUM"
        persona  = "HUB_LEVEL_ONLY"
        recommendation = "Conditional Dispatch: Package reaches delivery hub, but driver requires phone call for exact unit/door."
        success_rate = 70 if has_phone else 55
    elif pin_valid and not has_locality and not has_house:
        rto_risk = "HIGH"
        persona  = "AMBIGUOUS_ROUTE"
        recommendation = "Hold Dispatch: PIN code is valid, but address lacks specific street/area details."
        success_rate = 35
    else:
        rto_risk = "MEDIUM"
        persona  = "REVIEW_RECOMMENDED"
        recommendation = "Review Address: Partial delivery data. Verify contact phone before vehicle loading."
        success_rate = 50

    return {
        "rto_risk":               rto_risk,
        "deliverability_tier":    persona,
        "estimated_success_rate": success_rate,
        "dispatch_action":        recommendation,
        "checklist":              checklist,
        "checklist_passed":       passed_count,
        "checklist_total":        len(checklist)
    }

