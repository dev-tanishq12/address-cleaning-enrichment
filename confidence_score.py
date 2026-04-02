"""
confidence_score.py
-------------------
Calculates a confidence score (0-100) for a cleaned address.

TWO MODES:
  1. Rule-based  — your original approach, always available
  2. ML-based    — RandomForest trained on synthetic data (requires scikit-learn)

WHY ML IMPROVES ON RULE-BASED:
Your rule-based scorer has two problems:

Problem 1 — You pick the weights by hand
  pincode = 25 pts, city = 10 pts... why these numbers?
  You guessed them. A RandomForest learns the weights FROM DATA.
  It figures out which fields actually predict address quality.

Problem 2 — Double-counting
  In your code:
    if district or state: score += 20   ← pincode validated
    if district:          score += 15   ← district found
  If district is present, BOTH conditions fire → district = 35 pts
  If only state is present → state = 20 + 15 = 35 pts too
  Both give the same score even though district is more specific.
  ML handles this automatically — it learns the true contribution
  of each field independently.

WHAT IS A RANDOMFOREST?
  - You give it many examples: (features → label)
  - Features = which fields are present in the address
  - Label    = the confidence score for that address
  - It builds many decision trees, each trained on a random subset
  - Final prediction = average of all trees
  - This is called "ensemble learning" — many weak learners → one strong one

TRAINING DATA:
  We don't have real labelled addresses, so we generate synthetic ones.
  We create random combinations of field presence (0/1) and compute
  the rule-based score as the label. The ML model then learns to
  predict this score from features — but discovers better weights
  than the ones we hard-coded.
"""

import os
import pickle
import numpy as np

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "confidence_model.pkl")


# ---------------------------------------------------------------
# RULE-BASED SCORER (your original, cleaned up)
# ---------------------------------------------------------------
# This is the baseline. We use it to generate training labels
# for the ML model, and as a fallback if sklearn isn't installed.
#
# FIXED vs your original:
# Removed the double-counting issue where district triggered
# both the "pincode validated" condition AND the "district" condition.
# Now each field contributes independently.
#
# Field weights and why:
#   pincode present   = 25  (most reliable field — exact 6-digit match)
#   pincode validated = 20  (pincode found in India Post DB → district/state known)
#   district found    = 15  (from DB — reliable)
#   state found       = 10  (broad — less specific than district)
#   city found        = 10  (from NLP — slightly less reliable)
#   coords found      = 10  (lat/lon available → highest quality)
#   locality found    =  5  (sub-area — nice to have)
#   house no found    =  5  (extracted — shows address is specific)
#   Total possible    = 100

def rule_based_score(record):
    """  
    Compute additive confidence score from extracted fields.

    Args:
        record : dict with address fields (output of pipeline)

    Returns:
        int 0-100
    """
    score = 0

    # Pincode present (you found a 6-digit number)
    if record.get("pincode"):
        score += 25

    # Pincode validated (it exists in India Post DB → district + state known)
    # This is separate from just finding a pincode — validation adds confidence
    if record.get("district") and record.get("pincode"):
        score += 20

    # District found (came from pincode DB lookup)
    if record.get("district"):
        score += 15

    # State found
    if record.get("state"):
        score += 10

    # City found (from NLP parser)
    if record.get("city"):
        score += 10

    # Coordinates available
    if record.get("latitude") and record.get("longitude"):
        score += 10

    # Locality found (sector, block, nagar etc.)
    if record.get("locality"):
        score += 5

    # House number found
    if record.get("house_no"):
        score += 5

    return min(score, 100)


# ---------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------
# Converts an address record into a numeric vector for the ML model.
# Each feature is 0 or 1 (field absent or present).
#
# WHY BINARY FEATURES?
# The model doesn't care about the actual value of city ("Delhi" vs "Mumbai")
# — it only cares WHETHER city was found or not. Presence = 1, Absence = 0.
#
# The token count is an exception — it's a continuous feature.
# More tokens = more information in the address = likely higher quality.

FEATURE_NAMES = [
    "has_pincode",        # 0 or 1
    "pincode_validated",  # 0 or 1 (pincode + district both present)
    "has_district",       # 0 or 1
    "has_state",          # 0 or 1
    "has_city",           # 0 or 1
    "has_locality",       # 0 or 1
    "has_coords",         # 0 or 1
    "has_house_no",       # 0 or 1
    "token_count",        # int (number of words in normalized address)
]

def extract_features(record):
    """
    Convert address record to feature vector.

    Returns:
        numpy array of shape (9,)
    """
    has_pincode   = 1 if record.get("pincode")   else 0
    has_district  = 1 if record.get("district")  else 0
    has_state     = 1 if record.get("state")     else 0
    has_city      = 1 if record.get("city")      else 0
    has_locality  = 1 if record.get("locality")  else 0
    has_coords    = 1 if (record.get("latitude") and record.get("longitude")) else 0
    has_house_no  = 1 if record.get("house_no")  else 0
    pin_validated = 1 if (has_pincode and has_district) else 0
    token_count   = len(record.get("normalized_address", "").split())

    return np.array([
        has_pincode,
        pin_validated,
        has_district,
        has_state,
        has_city,
        has_locality,
        has_coords,
        has_house_no,
        token_count,
    ], dtype=float)


# ---------------------------------------------------------------
# SYNTHETIC TRAINING DATA GENERATOR
# ---------------------------------------------------------------
# WHY SYNTHETIC?
# We don't have thousands of real labelled addresses.
# So we generate random combinations of field presence (0/1),
# compute the rule-based score as the label, and train the model
# to learn those patterns.
#
# The model won't just memorize the rule-based formula — it will
# find patterns in combinations of features, which is more powerful.
# For example: "if pincode + district + city all present → very high confidence"
# even if no single field individually scores that high.

def generate_training_data(n=3000):
    """
    Generate n synthetic (features, label) pairs.

    Returns:
        X : numpy array (n, 9) — feature vectors
        y : numpy array (n,)   — confidence scores 0-100
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(n):
        # Random field presence with realistic probabilities
        # (addresses more often have state than locality, etc.)
        record = {
            "pincode":            rng.choice([None, "110078"],  p=[0.35, 0.65]),
            "district":           rng.choice([None, "Delhi"],   p=[0.40, 0.60]),
            "state":              rng.choice([None, "Delhi"],   p=[0.25, 0.75]),
            "city":               rng.choice([None, "Delhi"],   p=[0.30, 0.70]),
            "locality":           rng.choice([None, "Sector 17"], p=[0.55, 0.45]),
            "latitude":           rng.choice([None, 28.61],     p=[0.40, 0.60]),
            "longitude":          rng.choice([None, 77.03],     p=[0.40, 0.60]),
            "house_no":           rng.choice([None, "Flat 12"], p=[0.45, 0.55]),
            "normalized_address": " ".join(["word"] * rng.integers(3, 12)),
        }

        # Label = rule-based score + small noise
        # Noise forces the model to generalise rather than just memorise
        base  = rule_based_score(record)
        noise = int(rng.integers(-5, 6))   # -5 to +5 random noise
        label = max(0, min(100, base + noise))

        X.append(extract_features(record))
        y.append(label)

    return np.array(X), np.array(y)


# ---------------------------------------------------------------
# TRAIN ML MODEL
# ---------------------------------------------------------------
# RandomForestRegressor:
#   - "Regressor" because confidence is a number (0-100), not a class
#   - "RandomForest" = many decision trees, each trained on random subset
#   - n_estimators=100 means 100 trees — more trees = more stable predictions
#   - max_depth=6 limits tree depth — prevents overfitting
#
# Overfitting = model memorises training data but fails on new data
# max_depth controls this by limiting how specific each tree can get

def train_model(save=True):
    """
    Train RandomForest on synthetic data and optionally save to disk.

    Returns trained model.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        print("[confidence] scikit-learn not installed.")
        print("  Run: pip install scikit-learn")
        return None

    print("[confidence] Generating training data...")
    X, y = generate_training_data(n=3000)

    # 80% train, 20% test — test set tells us how well model generalises
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[confidence] Training on {len(X_train)} samples...")
    model = RandomForestRegressor(
        n_estimators=100,   # 100 decision trees
        max_depth=6,        # max 6 splits per tree
        random_state=42     # for reproducibility
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"[confidence] Test MAE: {mae:.2f} points")
    # MAE = Mean Absolute Error
    # e.g. MAE=3.2 means predictions are off by 3.2 points on average
    # For a 0-100 scale, anything under 5 is good

    # Feature importance — what the model learned
    print("[confidence] Feature importances (what matters most):")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {name:<20} {bar} {imp:.3f}")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"[confidence] Model saved to {MODEL_PATH}")

    return model


# ---------------------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------------------

def load_model():
    """Load saved model from disk. Returns None if not found."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------
# MAIN SCORING FUNCTION
# ---------------------------------------------------------------
# This is what pipeline.py will call.
# Tries ML model first, falls back to rule-based if not available.

_model       = None   # cached model — loaded once, reused
_model_ready = False  # flag so we only attempt loading once

def compute_confidence(record):
    """
    Compute confidence score for a processed address record.

    Tries ML model first. Falls back to rule-based if model not trained.

    Args:
        record : dict — full pipeline output with all address fields

    Returns:
        dict with:
            score  (int)   : 0-100 confidence
            method (str)   : 'ml' or 'rule'
    """
    global _model, _model_ready

    # Load model on first call
    if not _model_ready:
        _model       = load_model()
        _model_ready = True

    rule_score = rule_based_score(record)

    if _model is not None:
        features   = extract_features(record).reshape(1, -1)
        ml_score   = int(round(float(_model.predict(features)[0])))
        ml_score   = max(0, min(100, ml_score))
        return {"score": ml_score, "method": "ml"}
    else:
        return {"score": rule_score, "method": "rule"}


# ---------------------------------------------------------------
# TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: train the model (only needed once)
    print("Training ML confidence model...")
    train_model(save=True)
    print()

    # Step 2: test on sample addresses
    test_records = [
        {
            "name": "Complete address (all fields)",
            "normalized_address": "flat no 12 sector 17 dwarka delhi 110078",
            "pincode":   "110078",
            "district":  "South West Delhi",
            "state":     "Delhi",
            "city":      "New Delhi",
            "locality":  "Sector 17",
            "latitude":  28.56,
            "longitude": 77.05,
            "house_no":  "flat no 12",
        },
        {
            "name": "Partial address (city + state only)",
            "normalized_address": "h 34 bangalore",
            "pincode":   None,
            "district":  None,
            "state":     "Karnataka",
            "city":      "Bengaluru Urban",
            "locality":  None,
            "latitude":  None,
            "longitude": None,
            "house_no":  "h 34",
        },
        {
            "name": "Very incomplete address",
            "normalized_address": "near water tank village",
            "pincode":   None,
            "district":  None,
            "state":     None,
            "city":      None,
            "locality":  None,
            "latitude":  None,
            "longitude": None,
            "house_no":  None,
        },
    ]

    print("=" * 55)
    for rec in test_records:
        result     = compute_confidence(rec)
        rule       = rule_based_score(rec)
        tier       = "HIGH" if result['score'] >= 70 else ("MEDIUM" if result['score'] >= 40 else "LOW")
        print(f"Address : {rec['name']}")
        print(f"Rule    : {rule}")
        print(f"ML      : {result['score']}  ({result['method']})")
        print(f"Tier    : {tier}")
        print("-" * 55)