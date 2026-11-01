"""
confidence_score.py
-------------------
Calculates a confidence score (0-100) and confidence tier for a cleaned address.

Balanced ML Model (RandomForest) + Rule-Based Scorer:
- Rebalanced feature weighting that avoids over-penalizing addresses without pincodes
- Supports both direct postal validation and reverse-lookup inference
- Geocoordinate validation and completeness checks
"""

import os
import pickle
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "confidence_model.pkl")

FEATURE_NAMES = [
    "has_pincode",        # 1 if directly provided by user
    "pincode_inferred",   # 1 if inferred via reverse lookup
    "pincode_validated",  # 1 if validated in India Post DB
    "has_district",       # 1 if district identified
    "has_state",          # 1 if state identified
    "has_city",           # 1 if city identified
    "has_locality",       # 1 if locality identified
    "has_coords",         # 1 if lat/lon available
    "has_house_no",       # 1 if house/unit number identified
    "token_count",        # total words in normalized address
]


# ---------------------------------------------------------------
# RULE-BASED SCORER (BALANCED)
# ---------------------------------------------------------------
def rule_based_score(record):
    """
    Compute balanced additive confidence score from extracted fields.
    Maximum possible score: 100.
    """
    score = 0

    # 1. House / Building / Flat number (15 pts) - indicates high delivery specificity
    if record.get("house_no"):
        score += 15

    # 2. Locality / Sector / Sub-area (15 pts) - neighborhood level resolution
    if record.get("locality"):
        score += 15

    # 2b. Landmark / POI (5 pts bonus for delivery routing)
    if record.get("landmark"):
        score += 5

    # 3. City / Town (15 pts)
    if record.get("city"):
        score += 15

    # 4. District (10 pts)
    if record.get("district"):
        score += 10

    # 5. State (10 pts)
    if record.get("state"):
        score += 10

    # 6. Geocoordinates (10 pts)
    if record.get("latitude") and record.get("longitude"):
        score += 10

    # 7. Postal Code (up to 25 pts)
    pincode = record.get("pincode")
    pincode_source = record.get("pincode_source", "provided" if pincode else None)
    is_validated = record.get("pincode_validated", True if pincode else False)

    if pincode:
        if pincode_source == "provided" and is_validated:
            score += 25
        elif pincode_source == "inferred":
            score += 18
        else:
            score += 10

    # Penalties & Discrepancies
    discrepancy = record.get("discrepancy") or {}
    if discrepancy.get("has_conflict"):
        sev = discrepancy.get("severity", "")
        if sev == "CRITICAL":
            score -= 35
        elif sev == "HIGH":
            score -= 25
        elif sev == "WARNING":
            score -= 15

    if record.get("pincode_error"):
        score -= 20

    return max(0, min(100, score))


# ---------------------------------------------------------------
# FEATURE VECTOR EXTRACTION
# ---------------------------------------------------------------
def extract_features(record):
    """Convert address record into a numeric feature vector."""
    pincode_source    = record.get("pincode_source", "")
    has_pincode       = 1 if (record.get("pincode") and pincode_source != "inferred") else 0
    pincode_inferred  = 1 if (record.get("pincode") and pincode_source == "inferred") else 0
    pincode_validated = 1 if record.get("pincode_validated", False) else 0

    has_district = 1 if record.get("district") else 0
    has_state    = 1 if record.get("state")    else 0
    has_city     = 1 if record.get("city")     else 0
    has_locality = 1 if record.get("locality") else 0
    has_coords   = 1 if (record.get("latitude") and record.get("longitude")) else 0
    has_house_no = 1 if record.get("house_no") else 0

    norm_text   = record.get("normalized_address", "")
    token_count = len(norm_text.split()) if norm_text else 0

    return np.array([
        has_pincode,
        pincode_inferred,
        pincode_validated,
        has_district,
        has_state,
        has_city,
        has_locality,
        has_coords,
        has_house_no,
        token_count,
    ], dtype=float)


# ---------------------------------------------------------------
# SYNTHETIC DATA GENERATOR & MODEL TRAINING
# ---------------------------------------------------------------
def generate_training_data(n=5000):
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(n):
        has_pin = rng.choice([0, 1], p=[0.35, 0.65])
        pin_valid = has_pin if rng.random() > 0.05 else 0
        pin_inferred = 1 if (not has_pin and rng.random() > 0.25) else 0

        has_dist  = 1 if (pin_valid or pin_inferred or rng.random() > 0.35) else 0
        has_state = 1 if (has_dist or rng.random() > 0.2) else 0
        has_city  = 1 if (rng.random() > 0.2) else 0
        has_loc   = 1 if (rng.random() > 0.25) else 0
        has_coords= 1 if (pin_valid or pin_inferred or (has_dist and rng.random() > 0.3)) else 0
        has_house = 1 if (rng.random() > 0.35) else 0
        tokens    = int(rng.integers(2, 14))

        sim_record = {
            "house_no":           "house" if has_house else None,
            "locality":           "locality" if has_loc else None,
            "city":               "city" if has_city else None,
            "district":           "district" if has_dist else None,
            "state":              "state" if has_state else None,
            "latitude":           28.6 if has_coords else None,
            "longitude":          77.1 if has_coords else None,
            "pincode":            "110001" if (has_pin or pin_inferred) else None,
            "pincode_source":     "inferred" if pin_inferred else ("provided" if has_pin else None),
            "pincode_validated":  bool(pin_valid or pin_inferred),
            "normalized_address": " ".join(["token"] * tokens),
        }

        base_score = rule_based_score(sim_record)
        noise = int(rng.integers(-3, 4))
        target_score = max(0, min(100, base_score + noise))

        X.append(extract_features(sim_record))
        y.append(target_score)

    return np.array(X), np.array(y)


def train_model(save=True):
    from sklearn.ensemble import RandomForestRegressor
    X, y = generate_training_data(5000)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")

    return model


# ---------------------------------------------------------------
# RUNTIME SCORING
# ---------------------------------------------------------------
_model = None
_model_ready = False

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def compute_confidence(record):
    """
    Compute confidence score (0-100) and confidence tier for record.
    """
    global _model, _model_ready
    if not _model_ready:
        _model = load_model()
        _model_ready = True

    rule_score = rule_based_score(record)

    if _model is not None:
        feats = extract_features(record).reshape(1, -1)
        try:
            score = int(round(float(_model.predict(feats)[0])))
            score = max(0, min(100, score))
            method = "ml"
        except Exception:
            score = rule_score
            method = "rule"
    else:
        score = rule_score
        method = "rule"

    # Enforce penalty from discrepancy detector on final score
    discrepancy = record.get("discrepancy") or {}
    if discrepancy.get("has_conflict"):
        sev = discrepancy.get("severity", "")
        if sev == "CRITICAL":
            score = min(35, score - 35)
        elif sev == "HIGH":
            score = min(40, score - 25)
        elif sev == "WARNING":
            score = max(0, score - 15)
        score = max(0, score)

    if score >= 75:
        tier = "HIGH"
    elif score >= 45:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "score": score,
        "tier":  tier,
        "method": method,
    }


if __name__ == "__main__":
    print("Training updated confidence model...")
    train_model(save=True)
    print("Training complete.")
