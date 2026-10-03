import json

import pandas as pd

from reviewsense.config import get_settings
from reviewsense.data import load_reviews
from reviewsense.text_mining.analyzer import ReviewAnalyzer

print("star model:", get_settings().sentiment_model)
analyzer = ReviewAnalyzer()                      # loads all models once (a few seconds)

# --- one review
result = analyzer.analyze("Great tacos but the tables were sticky and we waited forever.")
print(json.dumps(result, indent=2))

# --- every review, summarised per business
report = analyzer.business_report(load_reviews())
for business, summary in report.items():
    print(business, summary["avg_predicted_stars"], "stars | aspects:", summary["aspects"])

# --- a table for a spreadsheet / dashboard
rows = [{"business": b, "aspect": a, **v} for b, s in report.items() for a, v in s["aspects"].items()]
pd.DataFrame(rows).to_csv("artifacts/aspect_report.csv", index=False)
print("saved artifacts/aspect_report.csv")