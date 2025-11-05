"""Streamlit UI for live evaluation of the lead-scoring model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import requests
import streamlit as st

from mlsys.training.pipeline import build_feature_matrix

DEFAULT_API_URL = "http://localhost:8000"


def _format_probabilities(probs: List[float]) -> pd.DataFrame:
    df = pd.DataFrame({"lead": range(1, len(probs) + 1), "probability": probs})
    df["rank"] = df["probability"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("probability", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Lead Scoring Evaluator", layout="wide")
    st.title("Lead Scoring Evaluator")

    st.sidebar.header("API Settings")
    api_url = st.sidebar.text_input("Inference API base URL", value=DEFAULT_API_URL)

    st.sidebar.markdown("---")
    st.sidebar.header("Sample Data")
    if st.sidebar.button("Load training sample"):
        X, _ = build_feature_matrix()
        st.session_state["uploaded_df"] = X.head(50).reset_index(drop=True)

    uploaded_file = st.file_uploader("Upload CSV of leads", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
    elif "uploaded_df" not in st.session_state:
        st.info("Upload a CSV or load sample data from the sidebar to begin.")
        return

    df = st.session_state["uploaded_df"]
    st.subheader("Input Preview")
    st.dataframe(df.head(20))

    if st.button("Score leads"):
        payload = {"leads": df.where(pd.notna(df), None).to_dict(orient="records")}
        try:
            response = requests.post(f"{api_url}/score", json=payload, timeout=30)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            return

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
            return

        data = response.json()
        probs = data.get("probabilities", [])
        result_df = _format_probabilities(probs)

        st.subheader("Predicted Probabilities")
        st.dataframe(result_df)

        st.subheader("Raw Response")
        st.code(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
