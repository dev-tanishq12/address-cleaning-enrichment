import io
import csv
import time

from flask import Flask, render_template, request, jsonify, Response
from pipeline import process_address
from gemini_parser import gemini_status

app = Flask(__name__)

# ── CSV batch upload constants ─────────────────────────────────────────────────
CSV_MAX_ROWS     = 500           # hard cap per upload
CSV_INPUT_COLUMN = "address"     # expected column header (case-insensitive)

# Columns written to the enriched output CSV (in order)
ENRICHED_COLUMNS = [
    "row_number",
    "raw_address",
    "cleaned_address",
    "house_no", "locality", "landmark",
    "city", "district", "state",
    "pincode", "pincode_source",
    "latitude", "longitude",
    "confidence_score", "confidence_tier",
    "rto_risk", "deliverability_tier", "estimated_success_rate",
    "discrepancy_status", "discrepancy_severity",
    "phone_number",
    "error",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _flatten_result(row_num: int, raw: str, result: dict) -> dict:
    """Map pipeline output dict → flat CSV row dict."""
    deliv = result.get("deliverability") or {}
    disc  = result.get("discrepancy")    or {}
    return {
        "row_number":             row_num,
        "raw_address":            raw,
        "cleaned_address":        result.get("normalized_address", ""),
        "house_no":               result.get("house_no", ""),
        "locality":               result.get("locality", ""),
        "landmark":               result.get("landmark", ""),
        "city":                   result.get("city", ""),
        "district":               result.get("district", ""),
        "state":                  result.get("state", ""),
        "pincode":                result.get("pincode", ""),
        "pincode_source":         result.get("pincode_source", ""),
        "latitude":               result.get("latitude", ""),
        "longitude":              result.get("longitude", ""),
        "confidence_score":       result.get("confidence_score", ""),
        "confidence_tier":        result.get("confidence_tier", ""),
        "rto_risk":               deliv.get("rto_risk", ""),
        "deliverability_tier":    deliv.get("deliverability_tier", ""),
        "estimated_success_rate": deliv.get("estimated_success_rate", ""),
        "discrepancy_status":     disc.get("status", ""),
        "discrepancy_severity":   disc.get("severity", ""),
        "phone_number":           result.get("phone_number", ""),
        "error":                  "",
    }


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/clean", methods=["POST"])
def clean():
    """
    Single-address cleaning endpoint.

    Request body (JSON):
        { "address": "Flat no 12, Sec 17 Dwarka, Delhi - 110078" }

    Response (JSON):
        Full pipeline output dict with all fields + confidence score.
    """
    data = request.get_json()

    if not data or not data.get("address"):
        return jsonify({"error": "No address provided"}), 400

    address = data["address"].strip()
    if len(address) < 5:
        return jsonify({"error": "Address too short"}), 400

    try:
        result = process_address(address)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch():
    """
    JSON batch endpoint — process multiple addresses at once.

    Request body (JSON):
        { "addresses": ["addr1", "addr2", ...] }

    Response (JSON):
        { "results": [...], "summary": { high, medium, low, avg_score } }
    """
    data = request.get_json()

    if not data or not data.get("addresses"):
        return jsonify({"error": "No addresses provided"}), 400

    addresses = data["addresses"][:20]   # cap at 20 for performance
    results   = []

    for addr in addresses:
        if addr.strip():
            try:
                results.append(process_address(addr.strip()))
            except Exception as e:
                results.append({"raw_address": addr, "error": str(e)})

    scores = [r.get("confidence_score", 0) for r in results if "confidence_score" in r]
    summary = {
        "total":     len(results),
        "high":      sum(1 for r in results if r.get("confidence_tier") == "HIGH"),
        "medium":    sum(1 for r in results if r.get("confidence_tier") == "MEDIUM"),
        "low":       sum(1 for r in results if r.get("confidence_tier") == "LOW"),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
    }

    return jsonify({"results": results, "summary": summary})


@app.route("/upload", methods=["POST"])
def upload_csv():
    """
    CSV batch upload endpoint.

    Accepts a multipart/form-data POST with a CSV file field named 'file'.
    The CSV must contain a column named 'address' (case-insensitive).
    Up to CSV_MAX_ROWS rows are processed.

    Returns a downloadable enriched CSV with all pipeline fields appended.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Use field name 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400
    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    # ── read uploaded CSV ──────────────────────────────────────────────────────
    try:
        raw_bytes = uploaded.read().decode("utf-8-sig")   # strip BOM if present
    except UnicodeDecodeError:
        try:
            uploaded.seek(0)
            raw_bytes = uploaded.read().decode("latin-1")
        except Exception:
            return jsonify({"error": "Could not decode file. Use UTF-8 encoding."}), 400

    reader   = csv.DictReader(io.StringIO(raw_bytes))
    # normalise header names to lowercase
    original_fields = reader.fieldnames or []
    header_map      = {f.strip().lower(): f for f in original_fields}

    if CSV_INPUT_COLUMN not in header_map:
        return jsonify({
            "error": f"CSV must contain an '{CSV_INPUT_COLUMN}' column. "
                     f"Found columns: {list(original_fields)}"
        }), 400

    rows = list(reader)
    if len(rows) == 0:
        return jsonify({"error": "CSV file is empty (no data rows)."}), 400

    capped = rows[:CSV_MAX_ROWS]

    # ── process each row ───────────────────────────────────────────────────────
    output_rows = []
    total       = len(capped)
    addr_col    = header_map[CSV_INPUT_COLUMN]   # original case column name

    # Keep any extra columns the user supplied (pass them through)
    extra_cols = [f for f in original_fields if f.strip().lower() != CSV_INPUT_COLUMN]

    for i, row in enumerate(capped, start=1):
        raw_addr = (row.get(addr_col) or "").strip()

        out_row = {}
        # carry through any extra user columns first
        for col in extra_cols:
            out_row[col] = row.get(col, "")

        if not raw_addr:
            out_row.update({col: "" for col in ENRICHED_COLUMNS})
            out_row["row_number"] = i
            out_row["raw_address"] = ""
            out_row["error"] = "EMPTY_ADDRESS"
        else:
            try:
                result  = process_address(raw_addr)
                flat    = _flatten_result(i, raw_addr, result)
                out_row.update(flat)
            except Exception as e:
                out_row.update({col: "" for col in ENRICHED_COLUMNS})
                out_row["row_number"]  = i
                out_row["raw_address"] = raw_addr
                out_row["error"]       = str(e)

        output_rows.append(out_row)

    # ── write enriched CSV to buffer ───────────────────────────────────────────
    output_cols = extra_cols + ENRICHED_COLUMNS
    buf         = io.StringIO()
    writer      = csv.DictWriter(buf, fieldnames=output_cols, extrasaction="ignore",
                                 lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(output_rows)

    # ── metadata summary header comment (first line) ───────────────────────────
    ts         = time.strftime("%Y-%m-%d %H:%M:%S")
    high_count = sum(1 for r in output_rows if r.get("confidence_tier") == "HIGH")
    mid_count  = sum(1 for r in output_rows if r.get("confidence_tier") == "MEDIUM")
    low_count  = sum(1 for r in output_rows if r.get("confidence_tier") == "LOW")

    summary_comment = (
        f"# AddressIQ Enriched Export | processed:{total} | "
        f"high:{high_count} | medium:{mid_count} | low:{low_count} | ts:{ts}\r\n"
    )
    csv_body = summary_comment + buf.getvalue()

    # ── stream file back ───────────────────────────────────────────────────────
    filename = f"addressiq_enriched_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_body,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.route("/status", methods=["GET"])
def status():
    """
    Health & Gemini availability check endpoint.

    Response (JSON):
        { "status": "ok", "gemini_available": bool, "model": str }
    """
    gs = gemini_status()
    return jsonify({
        "status":           "ok",
        "gemini_available": gs["gemini_available"],
        "model":            gs["model"],
    })


if __name__ == "__main__":
    app.run(debug=True)