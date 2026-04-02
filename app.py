from flask import Flask, render_template, request, jsonify
from pipeline import process_address

app = Flask(__name__)

@app.route("/")
def home():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/clean", methods=["POST"])
def clean():
    """
    API endpoint — receives a raw address, returns cleaned structured data.

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
    Batch endpoint — process multiple addresses at once.

    Request body (JSON):
        { "addresses": ["addr1", "addr2", ...] }

    Response (JSON):
        { "results": [...], "summary": { high, medium, low, avg_score } }
    """
    data = request.get_json()

    if not data or not data.get("addresses"):
        return jsonify({"error": "No addresses provided"}), 400

    addresses = data["addresses"][:20]  # cap at 20 for performance
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


if __name__ == "__main__":
    app.run(debug=True)