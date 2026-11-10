# AddressIQ - Product Requirements Document

**Version:** 2.6 (Google GenAI SDK migration)  
**Updated:** November 10, 2026  
**Status:** In progress  
**Supersedes:** The supplied `AddressIQ_PRD` PDF, version 2.0, and this document's prior v2.4 revision as the editable product reference

This revision reviews every project-level Python source file: `app.py`, `confidence_score.py`, `deliverability_index.py`, `discrepancy_detector.py`, `eda.py`, `gemini_parser.py`, `nlp_parser.py`, `pipeline.py`, `pincode_lookup.py`, `rule_engine.py`, and `test_delhi.py`. "Implemented" means behavior is present in source; it does not imply production readiness or independently validated accuracy. Features not evidenced in the repository are identified as future work rather than current commitments.

## 1. Product summary

AddressIQ cleans and enriches free-form Indian postal addresses. It combines deterministic text extraction, optional Gemini parsing, a local India Post reference dataset, fuzzy NLP fallback, discrepancy checks, address confidence scoring, and a logistics-readiness assessment. Users can process one address in a web dashboard or API, submit a small JSON batch, or upload a CSV and download enriched rows.

The product helps users identify missing address details and possible postal/geographic conflicts before using an address. It does not verify that a person lives at an address, guarantee that a parcel will be delivered, or dispatch a shipment.

## 2. Users and primary jobs

- **Operations and logistics teams:** identify incomplete or contradictory addresses and prepare a standardized shipping label.
- **Data operations teams:** enrich address spreadsheets while preserving their existing non-address columns.
- **Integrators and developers:** submit addresses through the HTTP API or run the pipeline from the command line.
- **Project analysts:** inspect the India Post dataset and generate exploratory charts.

## 3. Product scope

### In scope

- Indian address normalization and extraction of address components.
- Postal PIN lookup and best-effort reverse inference using the local postal dataset.
- Geographic consistency alerts when a provided PIN is checked against address text.
- Confidence tiering and heuristic delivery-readiness guidance.
- Single-address and bounded batch workflows through the dashboard and HTTP API.
- Enriched CSV upload and download.
- Optional Gemini parsing with a local NLP fallback.

### Out of scope in the current implementation

- International address processing.
- Courier booking, tracking, live serviceability checks, or dispatch integrations.
- Identity, occupancy, or proof-of-address verification.
- Persistent customer/address records, user accounts, or multi-tenant workflows.
- A production-grade guarantee of delivery probability or calibrated fraud detection.

## 4. User workflows

### 4.1 Clean one address

1. A user enters an address in the dashboard or sends it to `POST /clean`.
2. The system extracts and normalizes address text, then parses components.
3. If configured and available, Gemini is used as the primary structured parser; otherwise, the NLP parser is used.
4. The system looks up a provided PIN or attempts a best-effort reverse lookup.
5. It returns structured fields, provenance, discrepancy information, confidence, and delivery-readiness guidance.
6. The dashboard presents missing fields, relevant warnings, coordinates when available, and a formatted label that can be copied or printed.

### 4.2 Process JSON batch

A client sends an array of addresses to `POST /batch`. The service processes at most the first 20 entries and returns per-address results with a total, confidence-tier counts, and average confidence score.

### 4.3 Enrich a CSV

1. A user selects or drops a `.csv` file in the dashboard, or posts a multipart file named `file` to `POST /upload`.
2. The input must contain an `address` column; header matching is case-insensitive.
3. The service processes up to 500 rows, preserves other input columns, and returns a downloadable enriched CSV.
4. Blank addresses and row-level processing failures are represented in the output instead of aborting the whole file.

### 4.4 Enable Gemini

Gemini is configured by the service operator, not by an address-checking user. The parser loads `GEMINI_API_KEY` from the repository-root `.env` file at startup; a process environment variable takes precedence. The populated `.env` file is excluded from Git. Keys must not be sent from the browser or exposed in the user interface. Restart the server after changing `.env`. If the key is absent or Gemini is unavailable, the pipeline uses the NLP fallback. `GET /status` reports application and Gemini availability; a successful clean response with `gemini_used: true` confirms that the model call succeeded for that request.

### 4.5 Run the pipeline or inspect postal data from the command line

- `pipeline.py --address "..."` processes one address; without `--address`, it processes seven built-in examples. Both modes print the current Gemini status and formatted results.
- `gemini_parser.py`, `rule_engine.py`, `nlp_parser.py`, and `pincode_lookup.py` each contain small interactive/manual example runners. The Gemini runner calls the external model when it is configured.
- `confidence_score.py` trains and saves a model when run directly.
- `eda.py` reads the postal CSV relative to the current working directory, prints dataset summaries, and generates five PNG charts in `eda_outputs/`.
- `test_delhi.py` is a one-off console demonstration of fuzzy matching for the misspelling "dlehi"; it contains no automated assertions and is not a comprehensive test suite.

## 5. Functional requirements

| ID | Requirement | Current behavior |
|---|---|---|
| FR-1 | Normalize address text and expand supported abbreviations. | Implemented in `rule_engine.py`; punctuation and spacing are normalized, and known Indian address abbreviations are expanded. |
| FR-2 | Extract house/unit number, six-digit PIN, Indian phone number, and landmark when recognizable. | Implemented with regular expressions in `rule_engine.py`; phone and landmark text are removed before downstream parsing. |
| FR-3 | Parse locality, city, and state. | Implemented with Gemini as an optional primary parser and fuzzy NLP as fallback. Gemini output is guarded by structured JSON parsing and field sanitization. |
| FR-4 | Enrich PIN, district, state, and coordinates from postal reference data. | Implemented using `data/india_post_pincode.csv`; direct lookup validates a supplied PIN. Reverse lookup can infer a candidate from locality/city/state signals. |
| FR-5 | Communicate PIN provenance and validation state. | Implemented through fields including `pincode_source`, `pincode_validated`, `pincode_error`, and enrichment notes. Inferred and directly provided PINs are distinct. |
| FR-6 | Detect geographic mismatch between address text and postal routing. | Implemented for provided PINs, including invalid PIN, state mismatch, and city/district mismatch results. Inferred PINs are not treated as an explicit user-provided PIN for this cross-check. |
| FR-7 | Return an address confidence score and tier. | Implemented as a 0-100 score with `HIGH` (75-100), `MEDIUM` (45-74), and `LOW` (0-44) tiers. A trained RandomForest is used when available; rule-based scoring is the fallback. |
| FR-8 | Assess delivery readiness and expose actionable missing factors. | Implemented as a checklist of unit, locality, landmark, validated PIN, phone, and coordinates, with a risk tier, persona, dispatch suggestion, and estimated success rate. |
| FR-9 | Provide single-address cleaning over HTTP. | Implemented at `POST /clean`. Empty input and addresses shorter than five characters are rejected. |
| FR-10 | Provide bounded JSON batch processing. | Implemented at `POST /batch`, capped at 20 input entries. |
| FR-11 | Accept and enrich bounded CSV uploads. | Implemented at `POST /upload`, capped at 500 rows. UTF-8 with BOM and Latin-1 decoding are supported. Extra columns are carried through. |
| FR-12 | Expose application and Gemini status. | Implemented at `GET /status`; reports service status, Gemini initialization availability, and model name when initialized. |
| FR-13 | Keep Gemini credentials on the server and separate from address-checking users. | The parser reads `GEMINI_API_KEY` from the backend process environment. The dashboard does not collect or transmit provider credentials. |
| FR-14 | Provide a browser dashboard for single input and CSV workflows. | Implemented in `templates/index.html`, including sample addresses, results, warnings, missing-field feedback, postal statistics, coordinates/map link, CSV download, and label copy/print actions. |
| FR-15 | Provide a command-line entry point for pipeline use. | Implemented in `pipeline.py` with `--address`; without it, the script runs the built-in sample addresses. |
| FR-16 | Support offline postal-data exploration. | Implemented as a developer utility in `eda.py`, which prints dataset summaries and generates five charts. This is not a user-facing API feature. |
| FR-17 | Resolve city/state and locality from the supplied Indian reference data. | Implemented in `nlp_parser.py` using exact aliases, token-window matching, RapidFuzz where installed, and a `difflib` fallback for official city/state matching. Locality extraction uses known patterns and then a stopword-filtered remainder of the address. |
| FR-18 | Report whether parsing and confidence scoring used their fallback paths. | Implemented through `parse_method`, `gemini_used`, and `scoring_method`; these describe the selected parser/scorer, not an accuracy guarantee. |
| FR-19 | Return reference postal dataset summaries and visualizations for project analysis. | Implemented only in `eda.py`: prints shape, types, missing values, unique counts, and office-type counts; saves charts for state counts, missing values, office types, geographic points, and unique PINs by state. |

## 6. Processing and enrichment rules

The current pipeline runs these steps in order:

1. **Rule extraction:** remove the first recognized Indian phone number, then the first recognized landmark clause; lowercase and normalize punctuation/spacing; expand dictionary abbreviations; extract a six-digit PIN and house/unit number; remove extracted PIN and house number from the text passed to NLP. PIN normalization supports embedded spaces or hyphens between the two three-digit groups.
2. **Structured parsing:** call Gemini with raw and normalized address text when an API key is configured. The rule-extracted phone and PIN take precedence; Gemini provides city/state/locality, may fill a missing house number and may supply landmark/phone when the rules did not. If Gemini is unavailable or parsing fails, parse normalized remaining text with the local fuzzy NLP parser.
3. **Postal enrichment:** directly look up a detected PIN in the local dataset. If there is no PIN, try reverse lookup using parsed locality/city/state. A matched post office may provide an inferred PIN, district, state, and coordinates. If no office is matched but city/district coordinates are known, district fallback can return coordinates and district/state without a PIN.
4. **Geographic completion:** if city exists and state is missing, resolve state by district lookup; if district is missing but city exists, use the city value as district.
5. **Discrepancy check:** compare a directly provided PIN against postal data. State and city/district conflicts are checked with a small set of explicit metro and historical-name allowances. Inferred PINs are not cross-validated by this stage; a district-coordinate fallback has no PIN to validate.
6. **Scoring and readiness:** calculate confidence and its tier, then apply delivery-readiness rules to the completed record.

The first matching PIN record in the CSV is used for direct lookup. Reference data and reverse-search indexes are loaded and built at module import time. City/district and locality are not independently verified against a second authoritative source.

## 7. Output contract

The single-address pipeline returns a structured record with these principal fields:

- Input and normalization: `raw_address`, `normalized_address`, `gemini_cleaned_address`.
- Address components: `house_no`, `locality`, `landmark`, `phone_number`, `city`, `district`, `state`.
- Postal/geographic enrichment: `pincode`, `pincode_source`, `pincode_validated`, `pincode_error`, `latitude`, `longitude`.
- Validation and provenance: `discrepancy`, `enrichment_notes`, `parse_method`, `gemini_used`.
- Scoring and operations: `confidence_score`, `confidence_tier`, `scoring_method`, `deliverability`.

The `deliverability` object includes `rto_risk`, `deliverability_tier`, `estimated_success_rate`, `dispatch_action`, a six-factor `checklist`, and checklist counts. The JSON batch endpoint returns these records under `results` plus summary statistics. CSV output flattens selected fields into columns and includes an `error` column.

### 7.1 HTTP and CSV details

| Route | Request | Success response |
|---|---|---|
| `GET /` | No body. | Renders the dashboard. |
| `POST /clean` | JSON object with a non-empty `address` string of at least five characters. | One complete pipeline record as JSON. |
| `POST /batch` | JSON object with `addresses`; the first 20 entries are considered. | `results` array and `summary` containing `total`, `high`, `medium`, `low`, and `avg_score`. Blank strings are skipped. |
| `POST /upload` | Multipart form with a `.csv` file in field `file`; input requires an `address` header. | CSV attachment prefixed by a comment line containing processed/high/medium/low counts and timestamp. |
| `GET /status` | No body. | `status`, `gemini_available`, and `model`. |

The enriched CSV contains input columns other than `address`, followed by `row_number`, `raw_address`, `cleaned_address`, `house_no`, `locality`, `landmark`, `city`, `district`, `state`, `pincode`, `pincode_source`, `latitude`, `longitude`, `confidence_score`, `confidence_tier`, `rto_risk`, `deliverability_tier`, `estimated_success_rate`, `discrepancy_status`, `discrepancy_severity`, `phone_number`, and `error`. PIN validation, PIN error details, parser metadata, checklist contents, and enrichment notes are not exported as dedicated columns.

### 7.2 Confidence and delivery scoring semantics

- The rule scorer adds points for house/unit (15), locality (15), landmark (5), city (15), district (10), state (10), coordinates (10), and PIN (25 when directly provided and validated, 18 when inferred, otherwise 10). It subtracts penalties for discrepancies and an invalid-PIN error, and clamps the rule score to 0-100.
- When a readable `data/confidence_model.pkl` is available, a RandomForestRegressor predicts a score from ten features: direct PIN, inferred PIN, PIN validation, district, state, city, locality, coordinates, house number, and normalized token count. A missing, unreadable, or prediction-failing model uses the rule score.
- The training routine generates 5,000 deterministic synthetic records; target scores are rule-based scores with small random noise. These synthetic targets are not independently labeled address outcomes. The ML score is not a statistical probability of delivery, correctness, or fraud.
- Discrepancy penalties are applied after model/rule scoring: critical conflicts cap the result at 35 after subtracting 35; high conflicts cap at 40 after subtracting 25; warnings subtract 15. The rule scorer itself also subtracts discrepancy and PIN-error penalties, so the rule fallback path can penalize a conflict in both scoring steps.
- Delivery `estimated_success_rate` is a fixed heuristic selected by readiness branch (15, 25, 35, 50, 55, 70, 90, or 95 percent depending on the case/contact field); it is not learned from courier delivery outcomes. Critical conflict detection is specific to `CRITICAL` conflicts, so other conflict severities do not automatically select the critical-conflict branch.
- The readiness personas are `CRITICAL_CONFLICT`, `UNROUTABLE`, `LAST_MILE_READY`, `HUB_LEVEL_ONLY`, `AMBIGUOUS_ROUTE`, and `REVIEW_RECOMMENDED`. The checklist score is informational; the persona is selected by ordered rules over conflict severity, PIN validation/presence, house number, locality/landmark, and phone availability.

## 8. User interface requirements

The dashboard should:

- Clearly distinguish Gemini parsing from the NLP fallback.
- Distinguish a directly provided, validated PIN from an inferred or unverified PIN.
- Show address fields, missing-field feedback, mismatch messages, score/tier, and delivery checklist.
- Show coordinates and an external map link only when coordinates are available.
- Support CSV selection/drop, processing feedback, error messages, summary counts, and download.
- Offer a shipping-label preview, plain-text/JSON copy, and print action.

**Implementation note:** the feature-importance bars currently displayed in the dashboard are fixed values in the template, not importance values computed from the loaded model. They must not be presented as a per-model explanation until wired to actual model output.

## 9. Data, model, and service dependencies

- **Coverage:** India only. Postal enrichment depends on `data/india_post_pincode.csv`, with columns expected by the code for PIN, district, state name, office name, latitude, and longitude. The EDA script also expects office type. Dataset size and completeness depend on the supplied file.
- **Startup dependency:** `pincode_lookup.py` reads the CSV and builds indexes when imported. A missing file, unreadable CSV, or missing expected columns can therefore prevent the pipeline and Flask app from starting; there is no configured alternate data source.
- **Matching:** RapidFuzz is used when available. NLP matching has a `difflib` fallback, but fuzzy alias matching and the fuzzy locality-to-office strategy require RapidFuzz; exact token matches remain available. Postal reverse lookup indexes office-name tokens, filters candidates by district/state where possible, and selects the highest-ranked match. Its match confidence is returned by the lookup helper but is not included in the final pipeline record or confidence features. A district fallback assigns an internal match-confidence value of 60 but returns no PIN.
- **Confidence model:** `data/confidence_model.pkl` is optional at runtime. The training routine creates synthetic examples whose targets are derived from the rule-based score with small noise, using a 100-tree RandomForest with depth 8 and a fixed seed. The score is not independently validated.
- **Gemini:** the `gemini-3.8-flash` model is optional and requires a configured API key and network access. The current pipeline passes the raw address and normalized form to Gemini when enabled; inputs may contain personal data such as phone numbers. Deployments must disclose this external processing and obtain appropriate user approval.
- **Gemini configuration:** the parser loads `.env` using `python-dotenv`, with process environment variables taking precedence by default. Configure credentials in the ignored local `.env` file or deployment environment/secret manager. It is not a user-entered address-checking setting.
- **Gemini fallback details:** client initialization is lazy. Missing key, initialization errors, API errors, invalid response JSON, quota/rate limits, and authentication failures return the fallback schema and activate NLP parsing. Invalid credentials and quota/rate-limit responses disable further Gemini calls for the current process, preventing repeated failures for subsequent addresses. Other transient request errors and invalid JSON are retried once with a 0.5-second delay; after repeated request errors Gemini is disabled for the process. Restarting the server retries Gemini initialization. Responses are JSON-parsed and string, PIN, and phone fields are sanitized, but the code does not perform postal-data validation on fields returned by Gemini unless they flow through the later PIN lookup.
- **Gemini SDK/model lifecycle:** the implementation uses Google's supported `google-genai` SDK and configures `gemini-3.8-flash`. Local client construction does not verify successful authorization or model availability; a successful parse request (`gemini_used: true`) is the confirmation that the configured credential and model worked for that request.
- **EDA:** `eda.py` requires pandas, Matplotlib, and Seaborn and writes to a relative `eda_outputs/` directory, which is created if absent. Its data path is relative to the process working directory, unlike the script-relative path in `pincode_lookup.py`.
- **Persistence:** the application code has no address database or account store. Runtime behavior and logs of a deployment may still retain data and must be reviewed separately.

## 10. Non-functional requirements and release constraints

### Required product behavior

- The core workflow must remain usable without Gemini by falling back to NLP parsing.
- Missing, ambiguous, or inferred information must not be represented as definitively verified.
- Batch errors should be attributable to an individual row where possible.
- Postal data and model artifacts must be available at deployment time; the postal CSV is excluded from version control and must be supplied separately.

### Current release limitations

- The Flask entry point runs with `debug=True`; this is for local development and is not a production deployment configuration.
- The service has no application authentication, authorization, rate limiting, or persistent secrets management in the reviewed code.
- The single-address route returns caught processing exceptions as HTTP 500 JSON messages.
- The CSV route has a row cap but no explicit application-level uploaded-file size limit.
- JSON batches and CSV uploads are truncated to their first 20 or 500 entries respectively, without an explicit truncation warning. The CSV progress animation is client-side and does not report server-side progress.
- JSON batch validation assumes `addresses` is a sliceable collection of strings; malformed types/items are not consistently rejected as a client error. Blank strings are skipped and are not represented in the result list. The single-address route expects a string because it calls `.strip()`.
- CSV parsing loads the complete uploaded content and all rows into memory before applying the 500-row cap. Row-level processing exceptions are written as text to the export; the implementation does not validate header uniqueness or impose an uploaded-byte limit.
- Reverse lookup is heuristic and can select a plausible-but-wrong post office for ambiguous localities.
- The confidence score is trained from synthetic, rule-derived labels. The delivery success percentage is assigned by rules. Neither is an empirically calibrated operational probability.
- Postal lookup has no error handling for a missing/malformed dataset at import time. A malformed or incompatible model is silently treated as unavailable by the loader.
- The project has no comprehensive automated product test suite in the reviewed folder. The standalone script `test_delhi.py` prints one misspelling comparison without assertions; the other executable examples are manual demos, not automated regression tests.

## 11. Acceptance criteria

The current implementation is considered functionally aligned when:

1. A valid single address returns the documented structured fields and identifies the parser/scoring method used.
2. With no usable Gemini configuration, single-address processing continues via NLP fallback.
3. A supplied valid PIN is labeled as provided and validated against the local dataset; an unsupported PIN is identified as unvalidated.
4. A locality without a supplied PIN may return an inferred PIN, and the result exposes that provenance rather than labeling it as user-provided.
5. Geographic conflicts for supplied PINs include a status, severity, explanation, and suggested resolution when available.
6. Confidence tiers follow the documented thresholds and delivery readiness includes all six checklist factors.
7. JSON batch processing does not process more than 20 entries.
8. CSV processing accepts an `address` header regardless of case, preserves other columns, processes no more than 500 rows, and returns row-level error information.
9. The dashboard makes conflicts, missing fields, inferred postal data, parser mode, and CSV processing errors visible to the user.
10. With no matched post office but a known district, the pipeline may return district/state/coordinates without claiming a PIN was inferred.
11. Scoring output identifies ML versus rule scoring, and neither confidence nor delivery estimates are represented as empirically verified probabilities.
12. The end-user dashboard does not request, store, or transmit a Gemini API key; Gemini credentials are configured only in the backend environment.

These are functional checks, not accuracy or service-level targets. Before production use, the product owner must define and validate separate accuracy, latency, availability, and privacy targets using representative, independently labeled data.

## 12. Prioritized follow-up

1. **Production readiness:** disable debug mode, define deployment configuration, add appropriate authentication/authorization and rate limits, and impose an upload-size limit.
2. **Data protection:** document retention/logging, add consent/disclosure for Gemini processing, and use managed secret storage in hosted deployments.
3. **Validation quality:** evaluate parsing, PIN inference, discrepancy detection, confidence tiers, and delivery estimates against a representative labeled dataset; calibrate or rename probability-like outputs.
4. **Inference transparency:** expose the source and uncertainty of reverse-lookup results and provide a user-confirmation path for ambiguous matches.
5. **Batch-limit clarity:** reject oversized requests or clearly report how many JSON entries/CSV rows were omitted; connect progress UI to real server-side progress or label it as indeterminate.
6. **Model explainability:** replace the dashboard's fixed feature-importance bars with values from the actual model, or remove the visualization.
7. **Input/data robustness:** validate JSON shapes and types, avoid loading an unbounded upload into memory, define header/encoding behavior, and provide actionable startup diagnostics when reference data is absent or invalid.
8. **Reference-data operations:** document dataset source, version/date, refresh procedure, and behavior when the data file is missing or stale.
9. **Quality assurance:** add automated unit and API tests covering normal, missing, conflicting, malformed, batch-limit, CSV, postal fallback, model fallback, and Gemini-fallback cases.

## 13. Revision history

| Version | Date | Change |
|---|---|---|
| 2.0 | November 10, 2026 | Supplied PRD PDF. |
| 2.1 | November 10, 2026 | Reconciled the PRD against the repository; documented current API, CSV and dashboard behavior, parser fallback, data provenance, scoring limitations, and production follow-up. |
| 2.2 | November 10, 2026 | Reviewed all project-level Python files and clarified pipeline precedence, NLP and reverse-lookup strategies, district fallback, exact scoring behavior, CLI/EDA scripts, dataset startup dependency, malformed-input and upload limitations, and the absence of an automated test suite. |
| 2.3 | November 10, 2026 | Added support for Gemini authorization-key prefix `AQ.`, corrected the environment-variable lookup, and documented the legacy Gemini SDK/model verification limitation. |
| 2.4 | November 10, 2026 | Removed browser-based key entry and the runtime key-setting endpoint; moved credential configuration to the backend environment, clarified the end-user address-checking workflow, and removed the API-key-presence field from the status response. |
| 2.5 | November 10, 2026 | Added startup loading of a Git-ignored `.env` file for `GEMINI_API_KEY`, retained environment-variable precedence, and documented local setup. |
