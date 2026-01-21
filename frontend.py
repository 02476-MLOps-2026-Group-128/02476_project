from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
FEATURE_SET_PATH = BASE_DIR / "configs" / "feature_sets" / "feature_set1.json"
STATS_PATH = BASE_DIR / "data" / "processed" / "standardization_params.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
API_PATH = "/predict/diagnosed_diabetes/MLP/feature_set1/"

DEFAULT_FEATURES = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_postprandial",
    "insulin_level",
    "gender_male",
    "gender_other",
    "ethnicity_black",
    "ethnicity_hispanic",
    "ethnicity_other",
    "ethnicity_white",
    "education_level_highschool",
    "education_level_no_formal",
    "education_level_postgraduate",
    "income_level_low",
    "income_level_lower-middle",
    "income_level_middle",
    "income_level_upper-middle",
    "employment_status_retired",
    "employment_status_student",
    "employment_status_unemployed",
    "smoking_status_former",
    "smoking_status_never",
]

FALLBACK_NUMERIC_STATS = {
    "age": {"mean": 50.0965, "std": 15.593308},
    "alcohol_consumption_per_week": {"mean": 2.0030375, "std": 1.418874},
    "physical_activity_minutes_per_week": {"mean": 119.08965, "std": 84.76467},
    "diet_score": {"mean": 5.99385, "std": 1.7819422},
    "sleep_hours_per_day": {"mean": 6.999059, "std": 1.0947185},
    "screen_time_hours_per_day": {"mean": 5.9943647, "std": 2.4693437},
    "family_history_diabetes": {"mean": 0.2194375, "std": 0.41386813},
    "hypertension_history": {"mean": 0.2519625, "std": 0.43414256},
    "cardiovascular_history": {"mean": 0.0792625, "std": 0.2701497},
    "bmi": {"mean": 25.606987, "std": 3.5894969},
    "waist_to_hip_ratio": {"mean": 0.8560336, "std": 0.04693122},
    "systolic_bp": {"mean": 115.79267, "std": 14.27097},
    "diastolic_bp": {"mean": 75.2262, "std": 8.202417},
    "heart_rate": {"mean": 69.61956, "std": 8.403044},
    "cholesterol_total": {"mean": 185.97113, "std": 32.00925},
    "hdl_cholesterol": {"mean": 54.0492, "std": 10.2607355},
    "ldl_cholesterol": {"mean": 102.9729, "std": 33.375965},
    "triglycerides": {"mean": 121.41988, "std": 43.405415},
    "glucose_postprandial": {"mean": 160.06563, "std": 30.91229},
    "insulin_level": {"mean": 9.0612755, "std": 4.951205},
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&display=swap');

:root {
  --bg-1: #f4f1e8;
  --bg-2: #e7f2ef;
  --text: #1f2937;
  --muted: #475569;
  --accent: #1f7a6b;
  --accent-2: #e07a5f;
  --card: rgba(255, 255, 255, 0.78);
  --border: rgba(15, 23, 42, 0.12);
}

html, body, [class*="css"] {
  font-family: "Source Serif 4", serif;
}

.stApp {
  background:
    radial-gradient(circle at 12% 18%, rgba(255, 241, 227, 0.75), transparent 46%),
    radial-gradient(circle at 88% 8%, rgba(229, 244, 255, 0.6), transparent 40%),
    linear-gradient(125deg, var(--bg-1), var(--bg-2));
  color: var(--text);
}

h1, h2, h3, h4, h5 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.02em;
}

#MainMenu, footer, header {
  visibility: hidden;
}

.hero {
  background: linear-gradient(135deg, rgba(31, 122, 107, 0.12), rgba(224, 122, 95, 0.12));
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 1.75rem 2rem;
  box-shadow: 0 24px 50px rgba(15, 23, 42, 0.08);
  animation: rise 0.8s ease-out both;
}

.hero-eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.72rem;
  color: var(--muted);
}

.hero-title {
  font-size: 2.2rem;
  margin: 0.3rem 0 0.6rem;
}

.hero-chip {
  display: inline-block;
  margin-top: 0.6rem;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(31, 122, 107, 0.3);
  font-size: 0.85rem;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.7);
}

div[data-testid="stForm"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 16px 32px rgba(15, 23, 42, 0.06);
}

div[data-testid="stForm"] > div {
  gap: 1.2rem;
}

button[kind="primary"] {
  background: var(--accent);
  border-radius: 14px;
  padding: 0.65rem 1.8rem;
  font-weight: 600;
}

button[kind="primary"]:hover {
  background: #18695d;
}

section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.7);
  border-right: 1px solid var(--border);
}

.result-card {
  border-radius: 22px;
  border: 1px solid var(--border);
  padding: 1.8rem;
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
  background: rgba(255, 255, 255, 0.85);
  animation: fade-in 0.7s ease-out both;
}

.result-title {
  font-size: 2rem;
  margin-bottom: 0.4rem;
}

.result-sub {
  color: var(--muted);
  font-size: 1rem;
}

.result-pill {
  display: inline-block;
  margin-top: 0.8rem;
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.95rem;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(14px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
"""


@dataclass(frozen=True)
class NumericField:
    """Configuration for a numeric input field."""

    key: str
    label: str
    unit: str
    min_value: float
    max_value: float
    step: float
    input_type: str
    allow_unknown: bool


@dataclass(frozen=True)
class CategoryField:
    """Configuration for a categorical input field."""

    name: str
    label: str
    options: list[str]
    feature_keys: list[str]
    mapping: dict[str, dict[str, float]]
    widget: str


NUMERIC_FIELDS = {
    "age": NumericField("age", "Age", "years", 0, 100, 1, "slider", False),
    "alcohol_consumption_per_week": NumericField(
        "alcohol_consumption_per_week", "Alcohol per week", "drinks/week", 0, 20, 0.5, "slider", False
    ),
    "physical_activity_minutes_per_week": NumericField(
        "physical_activity_minutes_per_week", "Activity per week", "min/week", 0, 600, 10, "slider", False
    ),
    "diet_score": NumericField("diet_score", "Diet score", "score", 0, 10, 0.1, "slider", False),
    "sleep_hours_per_day": NumericField(
        "sleep_hours_per_day", "Sleep per day", "hours/day", 0, 12, 0.1, "slider", False
    ),
    "screen_time_hours_per_day": NumericField(
        "screen_time_hours_per_day", "Screen time per day", "hours/day", 0, 16, 0.5, "slider", False
    ),
    "bmi": NumericField("bmi", "BMI", "kg/m2", 12, 45, 0.1, "number", False),
    "waist_to_hip_ratio": NumericField(
        "waist_to_hip_ratio", "Waist-to-hip ratio", "ratio", 0.6, 1.2, 0.01, "number", False
    ),
    "systolic_bp": NumericField("systolic_bp", "Systolic blood pressure", "mmHg", 80, 200, 1, "number", False),
    "diastolic_bp": NumericField("diastolic_bp", "Diastolic blood pressure", "mmHg", 50, 130, 1, "number", False),
    "heart_rate": NumericField("heart_rate", "Heart rate", "bpm", 40, 140, 1, "slider", False),
    "cholesterol_total": NumericField(
        "cholesterol_total", "Total cholesterol", "mg/dL", 100, 300, 1, "number", True
    ),
    "hdl_cholesterol": NumericField("hdl_cholesterol", "HDL cholesterol", "mg/dL", 20, 120, 1, "number", True),
    "ldl_cholesterol": NumericField("ldl_cholesterol", "LDL cholesterol", "mg/dL", 40, 250, 1, "number", True),
    "triglycerides": NumericField("triglycerides", "Triglycerides", "mg/dL", 50, 400, 1, "number", True),
    "glucose_postprandial": NumericField(
        "glucose_postprandial", "Postprandial glucose", "mg/dL", 80, 300, 1, "number", True
    ),
    "insulin_level": NumericField("insulin_level", "Insulin level", "uIU/mL", 2, 30, 0.1, "number", True),
}

GENDER_FIELD = CategoryField(
    name="gender",
    label="Gender",
    options=["Female", "Male", "Other"],
    feature_keys=["gender_male", "gender_other"],
    mapping={
        "Female": {"gender_male": 0.0, "gender_other": 0.0},
        "Male": {"gender_male": 1.0, "gender_other": 0.0},
        "Other": {"gender_male": 0.0, "gender_other": 1.0},
    },
    widget="radio",
)

ETHNICITY_FIELD = CategoryField(
    name="ethnicity",
    label="Ethnicity",
    options=["Asian", "Black", "Hispanic", "White", "Other"],
    feature_keys=["ethnicity_black", "ethnicity_hispanic", "ethnicity_other", "ethnicity_white"],
    mapping={
        "Asian": {
            "ethnicity_black": 0.0,
            "ethnicity_hispanic": 0.0,
            "ethnicity_other": 0.0,
            "ethnicity_white": 0.0,
        },
        "Black": {"ethnicity_black": 1.0},
        "Hispanic": {"ethnicity_hispanic": 1.0},
        "White": {"ethnicity_white": 1.0},
        "Other": {"ethnicity_other": 1.0},
    },
    widget="select",
)

EDUCATION_FIELD = CategoryField(
    name="education",
    label="Education",
    options=["Graduate", "High school", "No formal education", "Postgraduate", "I don't know"],
    feature_keys=["education_level_highschool", "education_level_no_formal", "education_level_postgraduate"],
    mapping={
        "Graduate": {
            "education_level_highschool": 0.0,
            "education_level_no_formal": 0.0,
            "education_level_postgraduate": 0.0,
        },
        "High school": {"education_level_highschool": 1.0},
        "No formal education": {"education_level_no_formal": 1.0},
        "Postgraduate": {"education_level_postgraduate": 1.0},
    },
    widget="select",
)

INCOME_FIELD = CategoryField(
    name="income",
    label="Income",
    options=["High", "Upper-middle", "Middle", "Lower-middle", "Low", "I don't know"],
    feature_keys=[
        "income_level_low",
        "income_level_lower-middle",
        "income_level_middle",
        "income_level_upper-middle",
    ],
    mapping={
        "High": {
            "income_level_low": 0.0,
            "income_level_lower-middle": 0.0,
            "income_level_middle": 0.0,
            "income_level_upper-middle": 0.0,
        },
        "Upper-middle": {"income_level_upper-middle": 1.0},
        "Middle": {"income_level_middle": 1.0},
        "Lower-middle": {"income_level_lower-middle": 1.0},
        "Low": {"income_level_low": 1.0},
    },
    widget="select",
)

EMPLOYMENT_FIELD = CategoryField(
    name="employment",
    label="Employment status",
    options=["Employed", "Retired", "Student", "Unemployed", "I don't know"],
    feature_keys=["employment_status_retired", "employment_status_student", "employment_status_unemployed"],
    mapping={
        "Employed": {
            "employment_status_retired": 0.0,
            "employment_status_student": 0.0,
            "employment_status_unemployed": 0.0,
        },
        "Retired": {"employment_status_retired": 1.0},
        "Student": {"employment_status_student": 1.0},
        "Unemployed": {"employment_status_unemployed": 1.0},
    },
    widget="select",
)

SMOKING_FIELD = CategoryField(
    name="smoking",
    label="Smoking status",
    options=["Current", "Former", "Never", "I don't know"],
    feature_keys=["smoking_status_former", "smoking_status_never"],
    mapping={
        "Current": {"smoking_status_former": 0.0, "smoking_status_never": 0.0},
        "Former": {"smoking_status_former": 1.0, "smoking_status_never": 0.0},
        "Never": {"smoking_status_former": 0.0, "smoking_status_never": 1.0},
    },
    widget="radio",
)

TRI_STATE_FIELDS = [
    ("family_history_diabetes", "Family history of diabetes"),
    ("hypertension_history", "Hypertension"),
    ("cardiovascular_history", "Cardiovascular disease"),
]


@st.cache_data
def load_feature_list(path_str: str) -> list[str]:
    """Load the feature list from disk.

    Args:
        path_str: Path to the feature set JSON file.

    Returns:
        List of feature names in the expected order.
    """
    path = Path(path_str)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return [str(item) for item in data]
        except (json.JSONDecodeError, OSError):
            return DEFAULT_FEATURES.copy()
    return DEFAULT_FEATURES.copy()


@st.cache_data
def load_numeric_stats(path_str: str) -> dict[str, dict[str, float]]:
    """Load normalization statistics for numeric features.

    Args:
        path_str: Path to the standardization parameters CSV.

    Returns:
        Mapping of feature -> mean/std.
    """
    stats = {key: value.copy() for key, value in FALLBACK_NUMERIC_STATS.items()}
    path = Path(path_str)
    if not path.exists():
        return stats
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                feature = row.get("feature")
                if not feature:
                    continue
                try:
                    mean = float(row.get("mean", "0"))
                    std = float(row.get("std", "1"))
                except ValueError:
                    continue
                stats[feature] = {"mean": mean, "std": std}
    except OSError:
        return stats
    return stats


@st.cache_data
def load_categorical_defaults(
    path_str: str,
    feature_list: list[str],
    numeric_features: set[str],
) -> dict[str, float]:
    """Compute default values for categorical features from training data.

    Args:
        path_str: Path to the processed training CSV.
        feature_list: Ordered list of all expected feature keys.
        numeric_features: Set of numeric feature keys.

    Returns:
        Mapping of categorical feature -> mean prevalence.
    """
    defaults = {feature: 0.0 for feature in feature_list if feature not in numeric_features}
    path = Path(path_str)
    if not path.exists():
        return defaults
    sums = {feature: 0.0 for feature in defaults}
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                count += 1
                for feature in sums:
                    value = row.get(feature)
                    if value is None:
                        continue
                    try:
                        sums[feature] += float(value)
                    except ValueError:
                        continue
    except OSError:
        return defaults
    if count == 0:
        return defaults
    return {feature: sums[feature] / count for feature in sums}


def apply_base_styles() -> None:
    """Inject the base theme styling."""
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)


def apply_result_styles(prediction: int | None) -> None:
    """Inject background styling based on the prediction outcome.

    Args:
        prediction: Model prediction (1 diabetes, 0 no diabetes).
    """
    if prediction is None:
        return
    if prediction == 1:
        background = "#fde8e8"
        accent = "#b42318"
    else:
        background = "#e8f7ee"
        accent = "#027a48"
    css = f"""
    .stApp {{
      background: {background};
    }}
    section[data-testid="stSidebar"] {{
      background: {background};
    }}
    .result-pill {{
      background: {accent};
      color: #ffffff;
    }}
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value within min and max bounds.

    Args:
        value: Value to clamp.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(min_value, min(value, max_value))


def align_to_step(value: float, min_value: float, step: float) -> float:
    """Align a value to a step grid.

    Args:
        value: Value to align.
        min_value: Minimum value for the grid.
        step: Step size.

    Returns:
        Aligned value.
    """
    if step <= 0:
        return value
    steps = round((value - min_value) / step)
    aligned = min_value + steps * step
    return round(aligned, 6)


def normalize_value(value: float, mean: float, std: float) -> float:
    """Normalize a value using mean and standard deviation.

    Args:
        value: Raw value.
        mean: Mean value.
        std: Standard deviation.

    Returns:
        Normalized value.
    """
    if std == 0:
        return 0.0
    return (value - mean) / std


def render_numeric_field(field: NumericField, stats: dict[str, dict[str, float]]) -> float:
    """Render a numeric input field.

    Args:
        field: Numeric field configuration.
        stats: Mapping of feature -> mean/std.

    Returns:
        Raw numeric value for the feature.
    """
    mean = stats.get(field.key, {}).get("mean", (field.min_value + field.max_value) / 2)
    default_value = clamp(mean, field.min_value, field.max_value)
    default_value = align_to_step(default_value, field.min_value, field.step)
    use_default = False
    if field.allow_unknown:
        use_default = st.checkbox(
            "I don't know (use average)",
            key=f"{field.key}_unknown",
        )
    label = field.label if not field.unit else f"{field.label} ({field.unit})"
    if field.input_type == "slider":
        value = st.slider(
            label,
            min_value=float(field.min_value),
            max_value=float(field.max_value),
            value=float(default_value),
            step=float(field.step),
            key=f"{field.key}_value",
            disabled=use_default,
        )
    else:
        value = st.number_input(
            label,
            min_value=float(field.min_value),
            max_value=float(field.max_value),
            value=float(default_value),
            step=float(field.step),
            key=f"{field.key}_value",
            disabled=use_default,
        )
    return float(mean) if use_default else float(value)


def render_tri_state(label: str, key: str) -> float | None:
    """Render a yes/no/unknown radio input.

    Args:
        label: Field label.
        key: Streamlit key.

    Returns:
        Raw value (1.0 for yes, 0.0 for no) or None for unknown.
    """
    choice = st.radio(
        label,
        ["Yes", "No", "I don't know"],
        horizontal=True,
        index=2,
        key=key,
    )
    if choice == "Yes":
        return 1.0
    if choice == "No":
        return 0.0
    return None


def encode_one_hot(
    selection: str,
    mapping: dict[str, dict[str, float]],
    feature_keys: list[str],
    defaults: dict[str, float],
) -> dict[str, float]:
    """Encode a categorical selection into one-hot style features.

    Args:
        selection: Selected option.
        mapping: Mapping of option -> feature values.
        feature_keys: Feature keys to emit.
        defaults: Default values for unknown selections.

    Returns:
        Dictionary of feature values for the category.
    """
    if selection in mapping:
        values = mapping[selection]
        return {key: float(values.get(key, 0.0)) for key in feature_keys}
    return {key: float(defaults.get(key, 0.0)) for key in feature_keys}


def render_category_field(field: CategoryField, defaults: dict[str, float]) -> dict[str, float]:
    """Render a categorical field and return encoded values.

    Args:
        field: Category field configuration.
        defaults: Default values for unknown selections.

    Returns:
        Dictionary of encoded feature values.
    """
    default_index = (
        field.options.index("I don't know") if "I don't know" in field.options else 0
    )
    if field.widget == "radio":
        selection = st.radio(
            field.label,
            field.options,
            index=default_index,
            horizontal=True,
            key=f"{field.name}_choice",
        )
    else:
        selection = st.selectbox(
            field.label,
            field.options,
            index=default_index,
            key=f"{field.name}_choice",
        )
    return encode_one_hot(selection, field.mapping, field.feature_keys, defaults)


def build_payload(
    feature_list: list[str],
    numeric_stats: dict[str, dict[str, float]],
    numeric_raw: dict[str, float],
    categorical_values: dict[str, float],
    categorical_defaults: dict[str, float],
    normalize_inputs: bool,
) -> dict[str, float]:
    """Construct the payload to send to the API.

    Args:
        feature_list: Ordered list of expected features.
        numeric_stats: Mapping of numeric feature stats.
        numeric_raw: Raw numeric values.
        categorical_values: One-hot categorical values.
        categorical_defaults: Defaults for categorical values.
        normalize_inputs: Whether to normalize numeric inputs.

    Returns:
        Feature payload with all expected keys.
    """
    payload: dict[str, float] = {}
    for feature in feature_list:
        if feature in numeric_stats:
            mean = numeric_stats[feature]["mean"]
            std = numeric_stats[feature]["std"]
            raw_value = numeric_raw.get(feature, mean)
            if normalize_inputs:
                payload[feature] = float(normalize_value(raw_value, mean, std))
            else:
                payload[feature] = float(raw_value)
        elif feature in categorical_values:
            payload[feature] = float(categorical_values[feature])
        else:
            payload[feature] = float(categorical_defaults.get(feature, 0.0))
    return payload


def post_prediction(backend_url: str, payload: dict[str, float]) -> tuple[dict[str, Any] | None, str | None]:
    """Send the payload to the prediction API.

    Args:
        backend_url: Base URL of the backend service.
        payload: Feature payload.

    Returns:
        Tuple of response JSON and error message.
    """
    url = f"{backend_url.rstrip('/')}{API_PATH}"
    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        return None, f"Could not reach the API: {exc}"
    if response.status_code != 200:
        detail = response.text
        try:
            data = response.json()
            detail = data.get("detail", detail)
        except ValueError:
            pass
        return None, f"API error ({response.status_code}): {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The API did not return valid JSON."


def render_result(result: dict[str, Any]) -> None:
    """Render the prediction result panel.

    Args:
        result: API response payload.
    """
    prediction = int(result.get("prediction", 0))
    probabilities = result.get("probabilities", {})
    prob_diabetes = float(probabilities.get("Diabetes", 0.0))
    prob_no = float(probabilities.get("No diabetes", 1.0 - prob_diabetes))
    prob_diabetes = clamp(prob_diabetes, 0.0, 1.0)
    prob_no = clamp(prob_no, 0.0, 1.0)

    if prediction == 1:
        title = "Diabetes detected"
        subtitle = "The model estimates a higher chance of diabetes."
        pill_text = f"Diabetes probability: {prob_diabetes * 100:.1f}%"
    else:
        title = "No diabetes detected"
        subtitle = "The model estimates a lower chance of diabetes."
        pill_text = f"No diabetes probability: {prob_no * 100:.1f}%"

    card = f"""
    <div class="result-card">
      <div class="result-title">{title}</div>
      <div class="result-sub">{subtitle}</div>
      <div class="result-pill">{pill_text}</div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)
    st.progress(prob_diabetes)
    st.caption(
        f"Diabetes probability: {prob_diabetes * 100:.1f}% | No diabetes probability: {prob_no * 100:.1f}%"
    )


def main() -> None:
    """Run the Streamlit frontend."""
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    apply_base_styles()

    feature_list = load_feature_list(str(FEATURE_SET_PATH))
    numeric_stats = load_numeric_stats(str(STATS_PATH))
    numeric_features = set(numeric_stats.keys())
    categorical_defaults = load_categorical_defaults(str(TRAIN_DATA_PATH), feature_list, numeric_features)

    backend_default = os.environ.get("BACKEND_URL") or os.environ.get("BACKEND") or "http://127.0.0.1:8000"
    st.sidebar.markdown("### Backend")
    backend_url = st.sidebar.text_input("Backend URL", value=backend_default)
    normalize_inputs = st.sidebar.checkbox("Normalize numeric inputs (recommended)", value=True)
    st.sidebar.caption("Endpoint: /predict/diagnosed_diabetes/MLP/feature_set1/")

    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Diabetes prediction</div>
          <div class="hero-title">Diabetes form</div>
          <div>Fill in the fields. Leave blank or choose "I don't know" to use averages.</div>
          <div class="hero-chip">Using feature_set1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None

    numeric_raw: dict[str, float] = {}
    categorical_values: dict[str, float] = {}

    with st.form("predict_form"):
        tabs = st.tabs(["Basics", "Lifestyle", "Health", "Vitals", "Labs", "Socioeconomic"])

        with tabs[0]:
            col_a, col_b = st.columns(2)
            with col_a:
                numeric_raw["age"] = render_numeric_field(NUMERIC_FIELDS["age"], numeric_stats)
            with col_b:
                categorical_values.update(render_category_field(GENDER_FIELD, categorical_defaults))
            categorical_values.update(render_category_field(ETHNICITY_FIELD, categorical_defaults))

        with tabs[1]:
            col_a, col_b = st.columns(2)
            with col_a:
                numeric_raw["alcohol_consumption_per_week"] = render_numeric_field(
                    NUMERIC_FIELDS["alcohol_consumption_per_week"],
                    numeric_stats,
                )
                numeric_raw["physical_activity_minutes_per_week"] = render_numeric_field(
                    NUMERIC_FIELDS["physical_activity_minutes_per_week"],
                    numeric_stats,
                )
                numeric_raw["diet_score"] = render_numeric_field(NUMERIC_FIELDS["diet_score"], numeric_stats)
            with col_b:
                numeric_raw["sleep_hours_per_day"] = render_numeric_field(
                    NUMERIC_FIELDS["sleep_hours_per_day"],
                    numeric_stats,
                )
                numeric_raw["screen_time_hours_per_day"] = render_numeric_field(
                    NUMERIC_FIELDS["screen_time_hours_per_day"],
                    numeric_stats,
                )
                categorical_values.update(render_category_field(SMOKING_FIELD, categorical_defaults))

        with tabs[2]:
            col_a, col_b = st.columns(2)
            with col_a:
                for feature_key, label in TRI_STATE_FIELDS:
                    value = render_tri_state(label, key=f"{feature_key}_tri")
                    mean = numeric_stats.get(feature_key, {}).get("mean", 0.0)
                    numeric_raw[feature_key] = mean if value is None else value
            with col_b:
                numeric_raw["bmi"] = render_numeric_field(NUMERIC_FIELDS["bmi"], numeric_stats)
                numeric_raw["waist_to_hip_ratio"] = render_numeric_field(
                    NUMERIC_FIELDS["waist_to_hip_ratio"],
                    numeric_stats,
                )

        with tabs[3]:
            col_a, col_b = st.columns(2)
            with col_a:
                numeric_raw["systolic_bp"] = render_numeric_field(NUMERIC_FIELDS["systolic_bp"], numeric_stats)
                numeric_raw["diastolic_bp"] = render_numeric_field(NUMERIC_FIELDS["diastolic_bp"], numeric_stats)
            with col_b:
                numeric_raw["heart_rate"] = render_numeric_field(NUMERIC_FIELDS["heart_rate"], numeric_stats)

        with tabs[4]:
            col_a, col_b = st.columns(2)
            with col_a:
                numeric_raw["cholesterol_total"] = render_numeric_field(
                    NUMERIC_FIELDS["cholesterol_total"],
                    numeric_stats,
                )
                numeric_raw["hdl_cholesterol"] = render_numeric_field(
                    NUMERIC_FIELDS["hdl_cholesterol"],
                    numeric_stats,
                )
                numeric_raw["ldl_cholesterol"] = render_numeric_field(
                    NUMERIC_FIELDS["ldl_cholesterol"],
                    numeric_stats,
                )
            with col_b:
                numeric_raw["triglycerides"] = render_numeric_field(
                    NUMERIC_FIELDS["triglycerides"],
                    numeric_stats,
                )
                numeric_raw["glucose_postprandial"] = render_numeric_field(
                    NUMERIC_FIELDS["glucose_postprandial"],
                    numeric_stats,
                )
                numeric_raw["insulin_level"] = render_numeric_field(
                    NUMERIC_FIELDS["insulin_level"],
                    numeric_stats,
                )

        with tabs[5]:
            col_a, col_b = st.columns(2)
            with col_a:
                categorical_values.update(render_category_field(EDUCATION_FIELD, categorical_defaults))
                categorical_values.update(render_category_field(INCOME_FIELD, categorical_defaults))
            with col_b:
                categorical_values.update(render_category_field(EMPLOYMENT_FIELD, categorical_defaults))

        submitted = st.form_submit_button("Submit prediction")

    if submitted:
        payload = build_payload(
            feature_list,
            numeric_stats,
            numeric_raw,
            categorical_values,
            categorical_defaults,
            normalize_inputs,
        )
        with st.spinner("Fetching model response..."):
            result, error = post_prediction(backend_url, payload)
        st.session_state["last_result"] = result
        st.session_state["last_error"] = error

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    result = st.session_state.get("last_result")
    if result:
        prediction_value = int(result.get("prediction", 0))
        apply_result_styles(prediction_value)
        render_result(result)


if __name__ == "__main__":
    main()
