import numpy as np
import pandas as pd
from pathlib import Path

# Analytics outputs are generated from simulated individual-level data for demonstration purposes only.
# This script transforms synthetic operational data into explainable impact analytics outputs.


def load_data():
    """Load and merge the synthetic youth datasets using youth_id."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    youth_df = pd.read_csv(data_dir / "youth.csv")
    learning_df = pd.read_csv(data_dir / "learning_activity.csv")
    skills_df = pd.read_csv(data_dir / "skills.csv")
    employment_df = pd.read_csv(data_dir / "employment.csv")

    merged = youth_df.merge(learning_df, on="youth_id", how="left")
    merged = merged.merge(skills_df, on="youth_id", how="left")
    merged = merged.merge(employment_df, on="youth_id", how="left")
    return merged


def calculate_opportunity_index(data):
    """Create a transparent Digital Opportunity Index for each youth record."""
    education_score_map = {"Primary": 25, "Secondary": 50, "Vocational": 75, "University": 100}
    data = data.copy()

    data["internet_access_score"] = data["internet_access"].astype(int)
    data["device_access_score"] = data["device_access"].astype(int)
    data["education_level_score"] = data["education_level"].map(education_score_map)

    data["digital_opportunity_score"] = np.round(
        0.30 * data["internet_access_score"] * 100
        + 0.25 * data["device_access_score"] * 100
        + 0.25 * data["digital_literacy"]
        + 0.20 * data["education_level_score"],
        2,
    )

    data["opportunity_level"] = np.select(
        [data["digital_opportunity_score"] >= 75, data["digital_opportunity_score"] >= 50],
        ["High Opportunity", "Medium Opportunity"],
        default="Low Opportunity",
    )

    output = data[["youth_id", "digital_opportunity_score", "opportunity_level"]]
    output_path = Path(__file__).resolve().parent.parent / "data" / "digital_opportunity_index.csv"
    output.to_csv(output_path, index=False)
    return output


def segment_youth(data):
    """Create youth segments based on digital readiness and learning engagement."""
    data = data.copy()

    engagement_score = np.clip(
        0.30 * data["completion_rate"]
        + 0.25 * np.clip(data["learning_hours"] / 80 * 100, 0, 100)
        + 0.25 * np.clip(data["login_frequency"] / 40 * 100, 0, 100)
        + 0.20 * np.clip((90 - data["last_activity_days"]) / 90 * 100, 0, 100),
        0,
        100,
    )

    digital_readiness_score = np.clip(
        0.35 * data["digital_literacy"]
        + 0.25 * data["internet_access"].astype(int) * 100
        + 0.20 * data["device_access"].astype(int) * 100
        + 0.20 * data["career_readiness_score"],
        0,
        100,
    )

    data["learning_engagement_score"] = np.round(engagement_score, 2)
    data["digital_readiness_score"] = np.round(digital_readiness_score, 2)

    data["youth_segment"] = np.select(
        [
            (data["digital_readiness_score"] >= 60) & (data["learning_engagement_score"] >= 50),
            (data["digital_readiness_score"] >= 35) & (data["learning_engagement_score"] >= 25),
        ],
        ["Digital Ready Youth", "Emerging Learners"],
        default="Digitally Excluded Youth",
    )

    output = data[["youth_id", "digital_readiness_score", "learning_engagement_score", "youth_segment"]]
    output_path = Path(__file__).resolve().parent.parent / "data" / "youth_segments.csv"
    output.to_csv(output_path, index=False)
    return output


def calculate_learning_risk(data):
    """Calculate engagement risk using transparent activity-based scoring logic."""
    data = data.copy()

    engagement_score = np.clip(
        0.30 * data["completion_rate"]
        + 0.25 * np.clip(data["learning_hours"] / 80 * 100, 0, 100)
        + 0.20 * np.clip(data["login_frequency"] / 40 * 100, 0, 100)
        + 0.25 * np.clip((90 - data["last_activity_days"]) / 90 * 100, 0, 100),
        0,
        100,
    )

    data["engagement_score"] = np.round(engagement_score, 2)

    lower_cutoff = np.percentile(data["engagement_score"], 25)
    upper_cutoff = np.percentile(data["engagement_score"], 75)

    data["risk_level"] = np.select(
        [data["engagement_score"] <= lower_cutoff, data["engagement_score"] >= upper_cutoff],
        ["High Risk", "Low Risk"],
        default="Medium Risk",
    )

    output = data[["youth_id", "engagement_score", "risk_level"]]
    output_path = Path(__file__).resolve().parent.parent / "data" / "learning_engagement_risk.csv"
    output.to_csv(output_path, index=False)
    return output


def main():
    """Run all analytics steps and save interpretive outputs."""
    data = load_data()
    calculate_opportunity_index(data)
    segment_youth(data)
    calculate_learning_risk(data)

    print("Analytics outputs created successfully.")
    print("Saved files:")
    print("- data/digital_opportunity_index.csv")
    print("- data/youth_segments.csv")
    print("- data/learning_engagement_risk.csv")


if __name__ == "__main__":
    main()
