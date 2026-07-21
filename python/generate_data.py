import numpy as np
import pandas as pd
from pathlib import Path
import uuid

# Individual-level data is simulated for demonstration purposes only.
# This script creates a hypothetical youth digital learning ecosystem for portfolio demonstration.

rng = np.random.default_rng(42)

n = 5000

regions = ["Yangon", "Mandalay", "Shan", "Ayeyarwady", "Other"]
program_cohorts = ["2024 Cohort", "2025 Cohort", "2026 Cohort"]
cohort_probabilities = [0.30, 0.35, 0.35]

education_levels = ["Primary", "Secondary", "Vocational", "University"]
genders = ["Female", "Male", "Non-binary", "Prefer not to say"]
gender_probabilities = [0.48, 0.47, 0.03, 0.02]

# Create three latent youth profiles so the synthetic population better reflects real-world segments.
profile_labels = ["Digital Ready Youth", "Emerging Learners", "Digitally Excluded Youth"]
profile_probabilities = [0.30, 0.45, 0.25]

youth_df = pd.DataFrame(
    {
        "youth_id": [str(uuid.uuid4()) for _ in range(n)],
        "age": rng.integers(15, 30, size=n),
        "gender": rng.choice(genders, size=n, p=gender_probabilities),
        "program_cohort": rng.choice(program_cohorts, size=n, p=cohort_probabilities),
        "youth_profile": rng.choice(profile_labels, size=n, p=profile_probabilities),
    }
)

# Profile-specific context shapes place-based access and education opportunities.
profile_to_urban_prob = {
    "Digital Ready Youth": 0.85,
    "Emerging Learners": 0.55,
    "Digitally Excluded Youth": 0.20,
}
profile_to_region_probs = {
    "Digital Ready Youth": [0.35, 0.30, 0.12, 0.10, 0.13],
    "Emerging Learners": [0.25, 0.22, 0.18, 0.18, 0.17],
    "Digitally Excluded Youth": [0.12, 0.12, 0.25, 0.25, 0.26],
}
profile_to_education_probs = {
    "Digital Ready Youth": [0.05, 0.25, 0.30, 0.40],
    "Emerging Learners": [0.20, 0.50, 0.20, 0.10],
    "Digitally Excluded Youth": [0.35, 0.45, 0.15, 0.05],
}

youth_df["urban_rural"] = np.array(
    ["Urban" if rng.random() < profile_to_urban_prob[profile] else "Rural" for profile in youth_df["youth_profile"]]
)
youth_df["region"] = np.array(
    [rng.choice(regions, p=profile_to_region_probs[profile]) for profile in youth_df["youth_profile"]]
)
youth_df["education_level"] = np.array(
    [rng.choice(education_levels, p=profile_to_education_probs[profile]) for profile in youth_df["youth_profile"]]
)

# Education influences digital literacy and future readiness.
education_rank = youth_df["education_level"].map({"Primary": 1, "Secondary": 2, "Vocational": 3, "University": 4})
urban_flag = (youth_df["urban_rural"] == "Urban").astype(int)
regional_access_flag = youth_df["region"].isin(["Yangon", "Mandalay"]).astype(int)

# Profile-based access levels create realistic digital opportunity differences.
profile_to_device_base = {
    "Digital Ready Youth": 0.85,
    "Emerging Learners": 0.55,
    "Digitally Excluded Youth": 0.20,
}
profile_to_internet_base = {
    "Digital Ready Youth": 0.82,
    "Emerging Learners": 0.50,
    "Digitally Excluded Youth": 0.15,
}

profile_index = youth_df["youth_profile"].map(profile_to_device_base)
device_prob = profile_index + 0.06 * urban_flag + 0.03 * regional_access_flag + 0.05 * (education_rank >= 3).astype(int)
device_prob = np.clip(device_prob, 0.10, 0.95)
youth_df["device_access"] = rng.random(n) < device_prob

device_access_int = youth_df["device_access"].astype(int)
profile_internet_index = youth_df["youth_profile"].map(profile_to_internet_base)
internet_prob = profile_internet_index + 0.08 * urban_flag + 0.05 * device_access_int + 0.04 * (education_rank >= 3).astype(int)
internet_prob = np.clip(internet_prob, 0.10, 0.98)
youth_df["internet_access"] = rng.random(n) < internet_prob

# Digital literacy is shaped by access, education, and profile context.
profile_to_literacy_base = {
    "Digital Ready Youth": 78,
    "Emerging Learners": 52,
    "Digitally Excluded Youth": 24,
}
profile_literacy_base = youth_df["youth_profile"].map(profile_to_literacy_base)
digital_literacy_base = profile_literacy_base + 6 * education_rank + 8 * device_access_int + 6 * youth_df["internet_access"].astype(int) + 3 * urban_flag
digital_literacy_noise = rng.normal(0, 5, size=n)
youth_df["digital_literacy"] = np.clip(np.round(digital_literacy_base + digital_literacy_noise), 0, 100).astype(int)

# Learning engagement reflects digital opportunity and learner motivation.
profile_to_engagement_base = {
    "Digital Ready Youth": 18,
    "Emerging Learners": 6,
    "Digitally Excluded Youth": -12,
}
profile_engagement_base = youth_df["youth_profile"].map(profile_to_engagement_base)
engagement_signal = (
    profile_engagement_base
    + 0.35 * youth_df["digital_literacy"]
    + 6 * education_rank
    + 7 * device_access_int
    + 6 * youth_df["internet_access"].astype(int)
    + 2 * urban_flag
    + rng.normal(0, 8, size=n)
)
engagement_signal = np.clip(engagement_signal, 0, 100)

# Create learner activity patterns that distinguish engaged and disengaged participants.
learning_activity_df = pd.DataFrame({"youth_id": youth_df["youth_id"]})
profile_to_activity_boost = {
    "Digital Ready Youth": 18,
    "Emerging Learners": 4,
    "Digitally Excluded Youth": -14,
}
profile_activity_boost = youth_df["youth_profile"].map(profile_to_activity_boost)

courses_enrolled_lambda = np.maximum(0.1, 2 + engagement_signal / 18 + profile_activity_boost / 10)
learning_activity_df["courses_enrolled"] = np.clip(rng.poisson(courses_enrolled_lambda), 0, 12).astype(int)

lessons_completed_lambda = np.maximum(0.1, 0.8 * learning_activity_df["courses_enrolled"] + 0.7 + engagement_signal / 24 + profile_activity_boost / 8)
login_frequency_lambda = np.maximum(0.1, 1.5 + engagement_signal / 20 + 0.6 * device_access_int + 0.5 * youth_df["internet_access"].astype(int) + profile_activity_boost / 8)

learning_activity_df["lessons_completed"] = np.clip(
    rng.poisson(lessons_completed_lambda), 0, 80
).astype(int)
learning_activity_df["learning_hours"] = np.clip(
    np.round(4 + engagement_signal / 5 + 0.5 * learning_activity_df["courses_enrolled"] + profile_activity_boost + rng.normal(0, 2.5, size=n)),
    0,
    80,
).astype(int)
learning_activity_df["login_frequency"] = np.clip(
    rng.poisson(login_frequency_lambda),
    0,
    40,
).astype(int)
learning_activity_df["last_activity_days"] = np.clip(
    np.round(80 - 0.55 * engagement_signal + 7 * (1 - device_access_int) + 6 * (1 - youth_df["internet_access"].astype(int)) + rng.normal(0, 7, size=n) - profile_activity_boost / 2),
    0,
    90,
).astype(int)
learning_activity_df["completion_rate"] = np.clip(
    np.round(
        45
        + 0.25 * engagement_signal
        + profile_activity_boost
        + 0.15 * learning_activity_df["learning_hours"]
        + rng.normal(0, 6, size=n)
    ),
    0,
    100,
).astype(int)

# Skills and readiness should be shaped by education, digital access, and engagement.
skills_df = pd.DataFrame({"youth_id": youth_df["youth_id"]})
profile_to_skill_base = {
    "Digital Ready Youth": 28,
    "Emerging Learners": 8,
    "Digitally Excluded Youth": -14,
}
profile_skill_base = youth_df["youth_profile"].map(profile_to_skill_base)
skills_df["digital_skill_score"] = np.clip(
    np.round(
        18
        + profile_skill_base
        + 0.35 * youth_df["digital_literacy"]
        + 0.25 * engagement_signal
        + 0.15 * learning_activity_df["completion_rate"]
        + rng.normal(0, 5, size=n)
    ),
    0,
    100,
).astype(int)
skills_df["communication_score"] = np.clip(
    np.round(
        22
        + 0.18 * engagement_signal
        + 0.12 * learning_activity_df["learning_hours"]
        + 2 * (education_rank >= 3).astype(int)
        + rng.normal(0, 4, size=n)
    ),
    0,
    100,
).astype(int)
skills_df["problem_solving_score"] = np.clip(
    np.round(
        20
        + 0.20 * engagement_signal
        + 0.10 * learning_activity_df["learning_hours"]
        + 2 * (education_rank >= 3).astype(int)
        + rng.normal(0, 4, size=n)
    ),
    0,
    100,
).astype(int)
skills_df["english_score"] = np.clip(
    np.round(
        20
        + 0.15 * engagement_signal
        + 0.10 * learning_activity_df["learning_hours"]
        + 4 * youth_df["internet_access"].astype(int)
        + 2 * (education_rank >= 3).astype(int)
        + rng.normal(0, 4, size=n)
    ),
    0,
    100,
).astype(int)
skills_df["career_readiness_score"] = np.clip(
    np.round(
        0.25 * skills_df["digital_skill_score"]
        + 0.20 * skills_df["communication_score"]
        + 0.20 * skills_df["problem_solving_score"]
        + 0.15 * skills_df["english_score"]
        + 0.20 * learning_activity_df["completion_rate"]
    ),
    0,
    100,
).astype(int)

# Employment outcomes should reflect the earlier impact pathway from access to skills.
employment_df = pd.DataFrame({"youth_id": youth_df["youth_id"]})
profile_to_employment_offset = {
    "Digital Ready Youth": 0.16,
    "Emerging Learners": 0.03,
    "Digitally Excluded Youth": -0.10,
}
profile_employment_offset = youth_df["youth_profile"].map(profile_to_employment_offset)

job_search_prob = 0.08 + 0.003 * skills_df["career_readiness_score"] + 0.0015 * learning_activity_df["completion_rate"] + 0.01 * (education_rank >= 3).astype(int) + profile_employment_offset
job_search_prob = np.clip(job_search_prob, 0.05, 0.95)
employment_df["job_search_status"] = np.where(
    rng.random(n) < job_search_prob,
    "Actively searching",
    "Not actively searching",
)

internship_prob = 0.06 + 0.0035 * skills_df["career_readiness_score"] + 0.001 * learning_activity_df["completion_rate"] + 0.02 * (education_rank >= 3).astype(int) + profile_employment_offset
internship_prob = np.clip(internship_prob, 0.05, 0.95)
employment_df["internship_completed"] = (rng.random(n) < internship_prob).astype(int)

placement_prob = (
    0.02
    + 0.005 * skills_df["career_readiness_score"]
    + 0.003 * skills_df["problem_solving_score"]
    + 0.002 * learning_activity_df["completion_rate"]
    + 0.02 * employment_df["internship_completed"]
    + 0.01 * (education_rank >= 3).astype(int)
    + profile_employment_offset
)
placement_prob = np.clip(placement_prob, 0.03, 0.95)
employment_df["job_placement"] = (rng.random(n) < placement_prob).astype(int)

income_band_values = []
for placement, readiness in zip(employment_df["job_placement"], skills_df["career_readiness_score"]):
    if placement == 1 and readiness >= 80:
        income_band_values.append("High")
    elif placement == 1 and readiness >= 60:
        income_band_values.append("Middle")
    elif readiness >= 45:
        income_band_values.append("Lower-middle")
    else:
        income_band_values.append("Low")

employment_df["income_band"] = income_band_values

# Individual-level data is simulated for demonstration purposes only.
# Save files to the data directory for downstream analysis and visualization.
data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(exist_ok=True)

youth_df[[
    "youth_id",
    "age",
    "gender",
    "region",
    "urban_rural",
    "program_cohort",
    "education_level",
    "device_access",
    "internet_access",
    "digital_literacy",
]].to_csv(data_dir / "youth.csv", index=False)

learning_activity_df[[
    "youth_id",
    "courses_enrolled",
    "lessons_completed",
    "learning_hours",
    "login_frequency",
    "last_activity_days",
    "completion_rate",
]].to_csv(data_dir / "learning_activity.csv", index=False)

skills_df[[
    "youth_id",
    "digital_skill_score",
    "communication_score",
    "problem_solving_score",
    "english_score",
    "career_readiness_score",
]].to_csv(data_dir / "skills.csv", index=False)

employment_df[[
    "youth_id",
    "job_search_status",
    "internship_completed",
    "job_placement",
    "income_band",
]].to_csv(data_dir / "employment.csv", index=False)

print("Synthetic datasets created successfully.")
print(f"Saved {len(youth_df)} youth records to {data_dir}")
