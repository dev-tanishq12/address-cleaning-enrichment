"""
discrepancy_detector.py
-----------------------
Postal Cross-Validation & Address Discrepancy Detector.
Protects e-commerce, logistics, and KYC systems against costly RTO (Return-to-Origin)
by detecting geographic contradictions between user text and official postal routing.
"""

def validate_address_discrepancy(user_state, user_city, pincode, pin_data):
    """
    Cross-validates user-provided geographic text against official India Post pincode data.

    Returns dict:
        has_conflict   (bool) : True if conflict/mismatch detected
        status         (str)  : 'VERIFIED', 'STATE_MISMATCH', 'CITY_MISMATCH', 'INVALID_PINCODE', 'NO_PINCODE'
        severity       (str)  : 'NONE', 'CRITICAL', 'WARNING', 'HIGH', 'INFO'
        message        (str)  : Plain English explanation
        expected       (str)  : What the pincode officially routes to
        found_in_text  (str)  : What the user typed
        suggestion     (str)  : Recommended resolution
    """
    if not pincode:
        return {
            "has_conflict": False,
            "status": "NO_PINCODE",
            "severity": "INFO",
            "message": "No direct postal code provided in text; inferred or estimated.",
            "expected": None,
            "found_in_text": None,
            "suggestion": "Include a 6-digit PIN code for definitive courier validation."
        }

    # Pincode was given but is completely missing from India Post database
    if not pin_data:
        return {
            "has_conflict": True,
            "status": "INVALID_PINCODE",
            "severity": "HIGH",
            "message": f"Pincode {pincode} does not exist in the official India Post postal database.",
            "expected": None,
            "found_in_text": pincode,
            "suggestion": "Double check the 6-digit postal code for typos."
        }

    pin_state = (pin_data.get("state") or "").strip().lower()
    pin_dist  = (pin_data.get("district") or "").strip().lower()

    # State normalization synonyms (official vs historical spellings)
    STATE_SYNONYMS = {
        "orissa": "odisha",
        "odisha": "odisha",
        "uttaranchal": "uttarakhand",
        "uttarakhand": "uttarakhand",
        "pondicherry": "puducherry",
        "puducherry": "puducherry",
        "delhi": "delhi",
        "new delhi": "delhi",
    }

    norm_user_state = STATE_SYNONYMS.get(user_state.lower().strip() if user_state else "", user_state.lower().strip() if user_state else "")
    norm_pin_state  = STATE_SYNONYMS.get(pin_state, pin_state)

    # -------------------------------------------------------------
    # 1. State Mismatch Check (CRITICAL)
    # -------------------------------------------------------------
    # E.g. User writes "Bangalore, Karnataka - 110001" (110001 = New Delhi)
    if norm_user_state and norm_pin_state:
        if norm_user_state != norm_pin_state and norm_user_state not in norm_pin_state and norm_pin_state not in norm_user_state:
            return {
                "has_conflict": True,
                "status": "STATE_MISMATCH",
                "severity": "CRITICAL",
                "message": f"Critical Conflict: Pincode {pincode} routes to {pin_data.get('state')}, but text indicates {user_state.title()}.",
                "expected": pin_data.get("state"),
                "found_in_text": user_state.title(),
                "suggestion": f"Check for postal code typo or change destination state to {pin_data.get('state')}."
            }

    # -------------------------------------------------------------
    # 2. City / District Mismatch Check (WARNING)
    # -------------------------------------------------------------
    # E.g. User writes "Pune - 400001" (400001 = Mumbai)
    if user_city and pin_dist:
        u_city = user_city.lower().strip()

        # Handle Greater Metropolitan Subdistricts
        is_metro_aligned = False
        if norm_pin_state == "delhi" and ("delhi" in u_city or u_city in ("delhi", "new delhi")):
            is_metro_aligned = True
        elif "mumbai" in u_city and "mumbai" in pin_dist:
            is_metro_aligned = True
        elif ("bengaluru" in u_city or "bangalore" in u_city) and "bengaluru" in pin_dist:
            is_metro_aligned = True
        elif ("gurgaon" in u_city or "gurugram" in u_city) and "gurugram" in pin_dist:
            is_metro_aligned = True
        elif ("noida" in u_city or "gautam buddha nagar" in u_city) and "gautam buddha nagar" in pin_dist:
            is_metro_aligned = True
        elif ("hyderabad" in u_city or "secunderabad" in u_city) and ("hyderabad" in pin_dist or "rangareddy" in pin_dist):
            is_metro_aligned = True
        elif ("kolkata" in u_city or "calcutta" in u_city) and ("kolkata" in pin_dist or "howrah" in pin_dist):
            is_metro_aligned = True

        if not is_metro_aligned:
            if u_city != pin_dist and u_city not in pin_dist and pin_dist not in u_city:
                return {
                    "has_conflict": True,
                    "status": "CITY_MISMATCH",
                    "severity": "WARNING",
                    "message": f"Geographic Discrepancy: Pincode {pincode} routes to {pin_data.get('district')} ({pin_data.get('state')}), but address mentions {user_city.title()}.",
                    "expected": pin_data.get("district"),
                    "found_in_text": user_city.title(),
                    "suggestion": f"Verify whether address belongs to {pin_data.get('district')} or if a different pincode was intended."
                }

    # -------------------------------------------------------------
    # 3. Verified & Consistent
    # -------------------------------------------------------------
    return {
        "has_conflict": False,
        "status": "VERIFIED",
        "severity": "NONE",
        "message": f"Pincode {pincode} is verified and consistent with {pin_data.get('district')}, {pin_data.get('state')}.",
        "expected": f"{pin_data.get('district')}, {pin_data.get('state')}",
        "found_in_text": f"{user_city or ''}, {user_state or ''}".strip(' ,'),
        "suggestion": "No discrepancy detected."
    }

