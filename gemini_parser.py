"""
gemini_parser.py
-----------------
Gemini AI-powered Indian address parser.

This module sends raw (or rule-normalized) address text to the Gemini API
and receives a structured JSON record with all address fields extracted.

Design principles:
  - Acts as the PRIMARY parser; nlp_parser.py becomes the FALLBACK
  - Returns the same field schema as nlp_parser.py for drop-in compatibility
  - Handles all Indian address challenges natively via LLM:
      abbreviations, transliteration, embedded phone/landmark, state-city inference
  - Zero hallucination guard: Gemini is instructed to return null for missing fields
  - Tracks parse_method = "gemini" or "fallback_nlp" so the UI can show a badge
  - Gracefully degrades if API key is absent or quota is exceeded
"""

import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Lazy-import google-genai so the app starts even without it ─────────────────
_client    = None
_generate_config = None
_api_ready = False
_MODEL_NAME = "gemini-3.8-flash"

def _init_gemini():
    """
    Initialise the Gemini client once, on first use.
    Reads GEMINI_API_KEY from environment variable.
    Returns True if ready, False if unavailable.
    """
    global _client, _generate_config, _api_ready

    if _api_ready:
        return True
    if _api_ready is None:        # already failed once
        return False

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[GeminiParser] GEMINI_API_KEY not set — using NLP fallback.")
        _api_ready = None          # mark as permanently unavailable this session
        return False

    try:
        from google import genai
        from google.genai import types

        _client = genai.Client(api_key=api_key)
        _generate_config = types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.95,
            max_output_tokens=512,
            response_mime_type="application/json",
        )
        _api_ready = True
        print(f"[GeminiParser] {_MODEL_NAME} client ready.")
        return True
    except Exception as e:
        print(f"[GeminiParser] Init failed: {e} — using NLP fallback.")
        _api_ready = None
        return False


# ── System prompt (sent once per model, cached by SDK) ───────────────────────
_SYSTEM_PROMPT = """
You are an expert Indian postal address parser. 
Your ONLY job is to extract structured fields from a raw Indian address string.

RULES:
1. Return ONLY valid JSON — no markdown, no explanation, no extra text.
2. Set a field to null if the information is not present or cannot be reliably inferred.
3. Do NOT guess or hallucinate pincodes, states, or cities that are not implied by the text.
4. Expand common Indian address abbreviations: blr→Bangalore, del/dilli→Delhi,
   mmbai/bom→Mumbai, hyd→Hyderabad, pnq→Pune, kol→Kolkata, maa/mas→Chennai,
   ggn/gurugram→Gurgaon, chd→Chandigarh, amd→Ahmedabad, ggn→Gurgaon,
   sec→Sector, soc→Society, appt/apts→Apartments, nagar→Nagar, rd→Road,
   st→Street, opp→Opposite, nr→Near, blk→Block, w→West, e→East, n→North, s→South.
5. For state abbreviations: up→Uttar Pradesh, mp→Madhya Pradesh, hp→Himachal Pradesh,
   uk→Uttarakhand, wb→West Bengal, tn→Tamil Nadu, ka/kar→Karnataka,
   mh→Maharashtra, rj→Rajasthan, ap→Andhra Pradesh, ts→Telangana, gj→Gujarat,
   pb→Punjab, hr→Haryana, br→Bihar, jk→Jammu and Kashmir.
6. phone_number: Extract Indian mobile numbers (10 digits, optionally starting with +91/0).
   Format as +91-XXXXXXXXXX. Return null if absent.
7. landmark: Extract only if explicitly introduced by: near, opp, opposite, behind,
   next to, adj to, beside, in front of, facing. Strip the trigger word.
8. house_no: Extract flat/plot/house/door/unit number. Include type prefix (Flat, Plot, H-, etc.)
9. locality: The sub-area, sector, block, colony, layout, extension, or society name.
10. pincode: Only if a 6-digit number is explicitly present in the text. Else null.

OUTPUT SCHEMA (return exactly this JSON structure):
{
  "house_no":    string | null,
  "locality":    string | null,
  "landmark":    string | null,
  "city":        string | null,
  "state":       string | null,
  "pincode":     string | null,
  "phone_number": string | null,
  "gemini_cleaned_address": string | null
}

gemini_cleaned_address: A clean, standardised version of the full address
  (Title Case, no abbreviations, no phone numbers, no trigger words like "near").
"""


# ── Main parse function ───────────────────────────────────────────────────────
def gemini_parse(raw_address: str, normalized_address: str = None) -> dict:
    """
    Parse an Indian address using the Google GenAI SDK.

    Args:
        raw_address:        The original user-typed address string.
        normalized_address: Rule-engine output (optional, helps Gemini).

    Returns:
        dict with keys: house_no, locality, landmark, city, state, pincode,
                        phone_number, gemini_cleaned_address, parse_method, gemini_used
        parse_method = "gemini"      → Gemini successfully parsed
        parse_method = "fallback_nlp" → API unavailable / errored, use nlp_parser
    """
    global _api_ready

    _FAIL = {
        "house_no":               None,
        "locality":               None,
        "landmark":               None,
        "city":                   None,
        "state":                  None,
        "pincode":                None,
        "phone_number":           None,
        "gemini_cleaned_address": None,
        "parse_method":           "fallback_nlp",
        "gemini_used":            False,
    }

    if not _init_gemini():
        return _FAIL

    # Build the user prompt — provide both raw and normalized for context
    input_text = raw_address.strip()
    if normalized_address and normalized_address.strip() != raw_address.strip():
        input_text = f"Raw: {raw_address.strip()}\nNormalized: {normalized_address.strip()}"

    prompt = f"Parse this Indian address:\n\n{input_text}"

    # ── Call Gemini with retry ────────────────────────────────────────────────
    for attempt in range(2):
        try:
            response = _client.models.generate_content(
                model=_MODEL_NAME,
                contents=_SYSTEM_PROMPT + "\n\n" + prompt,
                config=_generate_config,
            )
            raw_json = response.text.strip()

            # Strip markdown code fences if Gemini wraps response anyway
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json)

            parsed = json.loads(raw_json)

            # Sanitise — ensure all expected keys present
            result = {
                "house_no":               _clean(parsed.get("house_no")),
                "locality":               _clean(parsed.get("locality")),
                "landmark":               _clean(parsed.get("landmark")),
                "city":                   _clean(parsed.get("city")),
                "state":                  _clean(parsed.get("state")),
                "pincode":                _clean_pincode(parsed.get("pincode")),
                "phone_number":           _clean_phone(parsed.get("phone_number")),
                "gemini_cleaned_address": _clean(parsed.get("gemini_cleaned_address")),
                "parse_method":           "gemini",
                "gemini_used":            True,
            }
            return result

        except json.JSONDecodeError as e:
            print(f"[GeminiParser] JSON decode error (attempt {attempt+1}): {e}")
            if attempt < 1:
                time.sleep(0.5)
                continue
            return _FAIL

        except Exception as e:
            err_str = str(e).lower()
            if (
                "quota" in err_str
                or "rate" in err_str
                or "429" in err_str
                or "resource_exhausted" in err_str
            ):
                print(f"[GeminiParser] Rate limit hit — switching to NLP fallback.")
                _api_ready = None
                return _FAIL
            elif "api_key" in err_str or "401" in err_str or "403" in err_str:
                print(f"[GeminiParser] Invalid API key — switching to NLP fallback.")
                _api_ready = None
                return _FAIL
            else:
                print(f"[GeminiParser] Error (attempt {attempt+1}): {e}")
            if attempt < 1:
                time.sleep(0.5)
                continue
            _api_ready = None
            return _FAIL

    return _FAIL


# ── Sanitisers ────────────────────────────────────────────────────────────────
def _clean(val):
    """Strip whitespace, return None if empty."""
    if not val or not isinstance(val, str):
        return None
    val = val.strip()
    return val if val.lower() not in ("null", "none", "n/a", "", "-") else None


def _clean_pincode(val):
    """Return 6-digit pincode string or None."""
    if not val:
        return None
    digits = re.sub(r"\D", "", str(val))
    return digits if len(digits) == 6 else None


def _clean_phone(val):
    """Normalise to +91-XXXXXXXXXX or return None."""
    if not val:
        return None
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 10 and digits[0] in "6789":
        return f"+91-{digits}"
    return None


# ── Status helper (used by app.py for /status endpoint) ─────────────────────
def gemini_status() -> dict:
    """Return current Gemini availability for the /status API."""
    ready = _init_gemini()
    return {
        "gemini_available": ready,
        "model":            _MODEL_NAME if ready else None,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Flat no 12, Sec 17 Dwarka, near metro station, Delhi - 110078 call 9876543210",
        "H-34 kormangla 5th blk blr behind post office",
        "12/B Andheri w mmbai next to reliance fresh mob 9811223344",
        "Plot 9, opp shiva temple, noida up 201301",
        "sector-62 gurgaon haryana",
        "near water tank village up",
    ]

    print("=" * 65)
    print("  GEMINI ADDRESS PARSER — TEST SUITE")
    print("=" * 65)

    for addr in test_cases:
        print(f"\nINPUT : {addr}")
        result = gemini_parse(addr)
        method = result.get("parse_method", "?")
        print(f"METHOD: {method}")
        for k, v in result.items():
            if k not in ("parse_method", "gemini_used") and v:
                print(f"  {k:<26}: {v}")
