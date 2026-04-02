"""
pincode_lookup.py
-----------------
Functionality:
- Validate Indian pincodes
- Fetch district and state information
- Provide latitude and longitude as actual floats (not strings)
"""

import pandas as pd

DATA_PATH = "data/india_post_pincode.csv"

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

# WHY dtype={"pincode": str}?
# Pincodes like "011001" start with 0. If pandas reads them as integers,
# the leading zero is dropped and "011001" becomes 11001 — wrong length,
# breaks all lookups. Forcing str preserves the exact digits.

df = pd.read_csv(DATA_PATH, dtype={"pincode": str})

# -------------------------------------------------
# BASIC DATA CLEANING (ONE-TIME, runs when module loads)
# -------------------------------------------------

# Remove accidental spaces around pincode values
df['pincode'] = df['pincode'].str.strip()

# .str.title() converts "SOUTH WEST DELHI" → "South West Delhi"
# Makes output more readable. Applied once here so lookup is clean.
df['district']  = df['district'].str.title()
df['statename'] = df['statename'].str.title()

# Clean office names — strip suffixes like " B.O", " S.O", " H.O"
# B.O = Branch Office, S.O = Sub Office, H.O = Head Office
# These suffixes are metadata, not part of the actual place name.
# Useful later when we do NLP city matching against office names.
if 'officename' in df.columns:
    df['officename'] = (
        df['officename']
        .str.replace(r'\s+B\.O|\s+S\.O|\s+H\.O', '', regex=True)
        .str.title()
    )

# FIX: Convert lat/lon from string to float.
# WHY: The CSV stores them as strings (some rows have "NA" text).
# pd.to_numeric(..., errors='coerce') means:
#   - valid number string "28.61" → float 28.61
#   - invalid string "NA" or empty → NaN (not a number, but numeric type)
# Without this, any map plotting or distance math would crash.
df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')


# -------------------------------------------------
# PINCODE LOOKUP FUNCTION
# -------------------------------------------------

def lookup_pincode(pincode):
    """
    Look up enrichment data for a given 6-digit Indian pincode.

    Returns a dict with district, state, latitude, longitude.
    Returns None if pincode is invalid or not found in dataset.

    WHY these validation checks:
    - None check: pipeline may pass None if no pincode was extracted
    - isinstance check: catches integers accidentally passed in
    - isdigit + len check: rejects "11007X" or "1234" etc.
    """

    # Guard: missing or wrong type
    if pincode is None or not isinstance(pincode, str):
        return None

    # Guard: must be exactly 6 numeric digits
    if not pincode.isdigit() or len(pincode) != 6:
        return None

    # Filter dataset rows where pincode matches
    result = df[df['pincode'] == pincode]

    # Pincode not in dataset at all
    if result.empty:
        return None

    # WHY iloc[0]?
    # Multiple post offices can share the same pincode (e.g. a large area).
    # We just take the first row — district and state will be the same
    # for all rows with the same pincode anyway.
    row = result.iloc[0]

    # FIX: use row["col"] not row.get("col")
    # pandas Series does not have a safe .get() like dicts do.
    # Instead we read directly and handle NaN values explicitly.
    # float('nan') is truthy in Python — we convert NaN → None
    # so the pipeline can cleanly check `if result["latitude"]`.

    lat = row["latitude"]
    lon = row["longitude"]

    return {
        "pincode":   pincode,
        "district":  row["district"]  if pd.notna(row["district"])  else None,
        "state":     row["statename"] if pd.notna(row["statename"]) else None,
        "latitude":  float(lat)       if pd.notna(lat)              else None,
        "longitude": float(lon)       if pd.notna(lon)              else None,
    }

# -------------------------------------------------
# HELPER FUNCTIONS FOR NLP PARSER
# -------------------------------------------------

# Returns a list of all unique district names from the dataset
# The NLP parser uses this as its city matching list
def get_all_cities():
    cities = df['district'].dropna().unique().tolist()
    return sorted(cities)

# Returns a list of all unique state names from the dataset
def get_all_states():
    states = df['statename'].dropna().unique().tolist()
    return sorted(states)
# -------------------------------------------------
# TESTING
# -------------------------------------------------

if __name__ == "__main__":
    test_pincodes = [
        "110078",   # Delhi        → should have district + coords
        "201301",   # Noida        → should have all fields
        "560034",   # Bangalore    → should have all fields
        "999999",   # Invalid      → should return None
        None,       # Edge case    → should return None
        110078,     # Int, not str → should return None (isinstance check)
        "11007",    # Too short    → should return None (len check)
    ]

    for pin in test_pincodes:
        result = lookup_pincode(pin)
        print(f"Input    : {repr(pin)}")
        if result:
            print(f"District : {result['district']}")
            print(f"State    : {result['state']}")
            print(f"Lat/Lon  : {result['latitude']}, {result['longitude']}")
            # Show the TYPE to confirm lat/lon are floats now, not strings
            print(f"Lat type : {type(result['latitude'])}")
        else:
            print(f"Result   : None")
        print("-" * 60)