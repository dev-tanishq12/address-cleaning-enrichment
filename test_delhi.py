import difflib
from pincode_lookup import get_all_states

states = [s.lower() for s in get_all_states()]
print(f"Total states: {len(states)}")
print(f"Score of 'dlehi' vs 'delhi': {difflib.SequenceMatcher(None, 'dlehi', 'delhi').ratio() * 100:.2f}%")

best_match = None
best_score = 0
for s in states:
    score = difflib.SequenceMatcher(None, 'dlehi', s).ratio() * 100
    if score >= 80 and score > best_score:
        best_match = s
        best_score = score

print(f"Best match: {best_match} at {best_score:.2f}%")