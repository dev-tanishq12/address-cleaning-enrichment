# AddressIQ — AI-Powered Address Cleaning & Enrichment System

A hybrid ML + NLP + DS pipeline that transforms messy Indian addresses into structured, validated data.

## Features
- Rule-based text normalization and abbreviation expansion
- India Post pincode database lookup (165K records)
- Fuzzy NLP parser for city, state, and locality extraction
- RandomForest ML confidence scoring
- Flask web dashboard

## Project Structure
```
address_cleaning/
├── data/                    # Add india_post_pincode.csv here (see below)
├── templates/index.html     # Web dashboard
├── app.py                   # Flask server
├── rule_engine.py           # Text cleaning
├── pincode_lookup.py        # India Post DB lookup
├── nlp_parser.py            # Fuzzy NLP extraction
├── confidence_score.py      # ML confidence scoring
├── pipeline.py              # End-to-end pipeline
└── eda.py                   # Data analysis charts
```

## Dataset
Download the India Post Pincode dataset and place it at `data/india_post_pincode.csv`.

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python confidence_score.py   # trains ML model
python app.py                # starts web server
```

Open `http://127.0.0.1:5000` in your browser.

## Tech Stack
Python · Flask · pandas · scikit-learn · spaCy · difflib
```

---

**Step 3 — Create a `requirements.txt`**
```
flask
pandas
numpy
scikit-learn
matplotlib
seaborn
rapidfuzz