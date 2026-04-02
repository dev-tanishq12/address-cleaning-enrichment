
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------
   
# Where to save the output charts
# os.makedirs creates the folder if it doesn't exist yet
# exist_ok=True means "don't crash if the folder already exists"
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sns.set_theme gives all charts a clean grid background
# without this, charts use matplotlib's default plain white look
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

# dtype={"pincode": str}  → keeps leading zeros (e.g. "011001")
# Without this, pandas reads pincodes as integers and drops the zero
df = pd.read_csv("data/india_post_pincode.csv", dtype={"pincode": str})

# Convert lat/lon from strings to actual numbers
# errors='coerce' means: if a value can't be converted (like "NA"),
# replace it with NaN instead of crashing
df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Standardize text columns to Title Case
# "UTTAR PRADESH" → "Uttar Pradesh"
# "south west delhi" → "South West Delhi"
df['district']  = df['district'].str.title()
df['statename'] = df['statename'].str.title()


# ---------------------------------------------------------------
# STEP 1 — BASIC STATS (printed, not charted)
# ---------------------------------------------------------------

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

# .shape returns (rows, columns) as a tuple
print(f"Total rows    : {df.shape[0]:,}")   # :, adds comma formatting → 165,627
print(f"Total columns : {df.shape[1]}")

print("\n--- Column data types ---")
# .dtypes shows what pandas thinks each column is
# str columns show as 'object' in pandas
print(df.dtypes)

print("\n--- Missing values ---")
# .isnull() returns True/False for each cell
# .sum() counts the Trues per column
# dividing by len(df) gives the fraction, *100 gives percentage
missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)

# Combine into one readable table
# Only show columns that actually have missing values
missing_df = pd.DataFrame({
    "missing_count": missing_count,
    "missing_%":     missing_pct
})
print(missing_df[missing_df["missing_count"] > 0])

print("\n--- Unique value counts ---")
for col in ["statename", "district", "pincode", "officetype"]:
    print(f"  {col}: {df[col].nunique()} unique values")

print("\n--- Office type breakdown ---")
# .value_counts() counts occurrences of each unique value
# sorted by most frequent first
print(df['officetype'].value_counts())


# ---------------------------------------------------------------
# STEP 2 — CHART 1: Top 15 states by post office count
# ---------------------------------------------------------------
# WHAT: horizontal bar chart
# WHY: shows which states dominate the dataset
#      important because your pipeline will mostly see addresses
#      from these states

fig, ax = plt.subplots(figsize=(10, 6))
# figsize=(width, height) in inches

# .value_counts() counts rows per state, .head(15) takes top 15
top_states = df['statename'].value_counts().head(15)

sns.barplot(
    x=top_states.values,   # bar length = count
    y=top_states.index,    # state name on y-axis (horizontal bars)
    ax=ax,
    hue=top_states.index,  # needed in newer seaborn to avoid FutureWarning
    palette="Blues_r",     # _r = reversed, so most = darkest blue
    legend=False
)

ax.set_title("Top 15 States by Number of Post Offices", fontsize=14, pad=12)
ax.set_xlabel("Number of Post Offices")
ax.set_ylabel("State")

# Add count labels at end of each bar
for i, v in enumerate(top_states.values):
    ax.text(v + 100, i, f"{v:,}", va='center', fontsize=9)

plt.tight_layout()  # prevents labels from getting cut off
fig.savefig(os.path.join(OUTPUT_DIR, "chart1_states.png"), dpi=150)
plt.close(fig)      # close the figure to free memory
print("\nChart 1 saved: chart1_states.png")


# ---------------------------------------------------------------
# STEP 3 — CHART 2: Missing data percentage per column
# ---------------------------------------------------------------
# WHAT: horizontal bar chart of missing % per column
# WHY: tells you exactly where your pipeline needs defensive coding
#      (pd.notna() checks, fallback values, etc.)

missing_pct_nonzero = missing_pct[missing_pct > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=missing_pct_nonzero.values,
    y=missing_pct_nonzero.index,
    ax=ax,
    hue=missing_pct_nonzero.index,
    palette="Reds_r",
    legend=False
)

ax.set_title("Missing Data % per Column", fontsize=14, pad=12)
ax.set_xlabel("Missing %")
ax.set_ylabel("Column")

# Add percentage labels at end of each bar
for i, v in enumerate(missing_pct_nonzero.values):
    ax.text(v + 0.05, i, f"{v:.1f}%", va='center', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart2_missing.png"), dpi=150)
plt.close(fig)
print("Chart 2 saved: chart2_missing.png")


# ---------------------------------------------------------------
# STEP 4 — CHART 3: Office type distribution (pie chart)
# ---------------------------------------------------------------
# WHAT: pie chart showing BO vs PO vs HO split
# WHY: 85% of records are Branch Offices — small rural offices
#      This is why 7% have no coordinates (remote locations)

office_counts = df['officetype'].value_counts()

# Full names for the chart labels
# The CSV stores short codes — we expand them for readability
label_map = {
    "BO": "Branch Office",
    "PO": "Post Office",
    "HO": "Head Office"
}

fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(
    office_counts.values,
    labels=[label_map.get(l, l) for l in office_counts.index],
    # label_map.get(l, l) means: look up l in label_map,
    # if not found use l itself as the label
    autopct='%1.1f%%',          # show percentage on each slice
    colors=sns.color_palette("muted"),
    startangle=90               # start from top, not right
)

ax.set_title("Post Office Type Distribution", fontsize=14, pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart3_officetypes.png"), dpi=150)
plt.close(fig)
print("Chart 3 saved: chart3_officetypes.png")


# ---------------------------------------------------------------
# STEP 5 — CHART 4: Geographic scatter of all post offices
# ---------------------------------------------------------------
# WHAT: scatter plot using latitude/longitude as x/y coordinates
# WHY: every dot is one post office — together they draw the
#      shape of India with no map background needed
#      Dense areas = more post offices = more urban

# Drop rows where lat or lon is missing
geo = df.dropna(subset=['latitude', 'longitude'])

# Filter to India's approximate bounding box
# This removes any bad data points (wrong coordinates)
# India spans roughly: lat 6°N to 37°N, lon 68°E to 98°E
geo = geo[
    (geo['latitude'].between(6, 37)) &
    (geo['longitude'].between(68, 98))
]

print(f"\nPlotting {len(geo):,} post offices with valid coordinates...")

fig, ax = plt.subplots(figsize=(7, 9))
ax.scatter(
    geo['longitude'],   # x-axis = east-west position
    geo['latitude'],    # y-axis = north-south position
    s=0.3,              # dot size (very small — we have 150K+ dots)
    alpha=0.2,          # transparency — overlapping dots darken naturally
    c='steelblue'
)

ax.set_title("Post Office Locations Across India", fontsize=14, pad=12)
ax.set_xlabel("Longitude (East ↑)")
ax.set_ylabel("Latitude (North ↑)")
ax.set_facecolor("#f0f0f0")   # light gray background makes dots pop

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart4_geo.png"), dpi=150)
plt.close(fig)
print("Chart 4 saved: chart4_geo.png")


# ---------------------------------------------------------------
# STEP 6 — CHART 5: Unique pincodes per state (your exercise)
# ---------------------------------------------------------------
# WHAT: how many distinct pincodes each state has
# WHY: different from Chart 1 (post office count)
#      A state can have few post offices but many pincodes
#      or many post offices sharing few pincodes
#
# KEY CONCEPT: groupby + nunique
#   df.groupby('statename')     → split data into groups by state
#   ['pincode'].nunique()       → for each group, count distinct pincodes
#   .sort_values(ascending=False) → sort highest first
#   .head(15)                   → top 15 only

unique_pins = (
    df.groupby('statename')['pincode']
    .nunique()
    .sort_values(ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=unique_pins.values,
    y=unique_pins.index,
    ax=ax,
    hue=unique_pins.index,
    palette="Greens_r",
    legend=False
)

ax.set_title("Unique Pincodes per State (Top 15)", fontsize=14, pad=12)
ax.set_xlabel("Number of Unique Pincodes")
ax.set_ylabel("State")

for i, v in enumerate(unique_pins.values):
    ax.text(v + 10, i, f"{v:,}", va='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart5_unique_pincodes.png"), dpi=150)
plt.close(fig)
print("Chart 5 saved: chart5_unique_pincodes.png")


# ---------------------------------------------------------------
# DONE
# ---------------------------------------------------------------
print(f"\nAll charts saved to: {OUTPUT_DIR}/")
print("Open the PNG files in VS Code by clicking on them.")