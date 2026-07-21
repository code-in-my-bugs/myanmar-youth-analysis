import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

segments = pd.read_csv(DATA_DIR / "youth_segments.csv")
doi = pd.read_csv(DATA_DIR / "digital_opportunity_index.csv")
risk = pd.read_csv(DATA_DIR / "learning_engagement_risk.csv")


print("\nYouth Segments")
print("----------------")
print(segments["youth_segment"].value_counts())
print(
    segments["youth_segment"]
    .value_counts(normalize=True)
    .round(3)
)


print("\nOpportunity Levels")
print("----------------")
print(doi["opportunity_level"].value_counts())
print(
    doi["opportunity_level"]
    .value_counts(normalize=True)
    .round(3)
)


print("\nLearning Risk")
print("----------------")
print(risk["risk_level"].value_counts())
print(
    risk["risk_level"]
    .value_counts(normalize=True)
    .round(3)
)