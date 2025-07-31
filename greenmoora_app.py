
# greenmoora_app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="GreenMOORA: Big Data MCDM for Sustainability", layout="wide")

# --- Title and Description ---
st.title("🌱 GreenMOORA: Big Data MCDM for Sustainability and Innovation Impact")
st.markdown("""
GreenMOORA is a decision support tool using the **MOORA method** to rank alternatives
(such as innovations, regions, or projects) based on **sustainability and impact** criteria.

Built with **Streamlit** and powered by **Big Data and MCDM**, this system helps 
evaluate performance across multiple dimensions with clarity and transparency.
""")

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload Decision Matrix (CSV or Excel)", type=["csv", "xlsx"])

# --- Read Data ---
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.warning("Please upload a dataset to proceed. The first column should contain alternative names, and the remaining columns numeric criteria.")
    st.stop()


# --- Validate Data ---
if df.shape[1] < 3:
    st.error("Dataset must include at least 1 identifier column and 2+ numeric criteria.")
    st.stop()

# --- Extract data ---
alternative_names = df.iloc[:, 0]
criteria_matrix = df.iloc[:, 1:]

# --- Input Weights and Impacts ---
st.subheader("⚖️ Criteria Weights and Impacts")

num_criteria = criteria_matrix.shape[1]
default_weights = [round(1 / num_criteria, 2)] * num_criteria

weights_input = st.text_input("Enter weights (comma-separated, must sum to 1):", ",".join(map(str, default_weights)))
impacts_input = st.text_input("Enter impacts (+ for benefit, - for cost):", ",".join(['+'] * num_criteria))

# --- Parse weights and impacts ---
try:
    weights = list(map(float, weights_input.split(',')))
    impacts = [i.strip() for i in impacts_input.split(',')]

    if len(weights) != num_criteria or len(impacts) != num_criteria:
        st.error("Weights and impacts must match number of criteria.")
        st.stop()
    if not np.isclose(sum(weights), 1.0):
        st.error("Weights must sum to 1.")
        st.stop()
    if not all(i in ['+', '-'] for i in impacts):
        st.error("Impacts must be '+' or '-' only.")
        st.stop()
except:
    st.error("Error parsing weights or impacts.")
    st.stop()

# --- Normalize matrix (Vector normalization) ---
normalized_matrix = criteria_matrix / np.sqrt((criteria_matrix ** 2).sum())
st.subheader("📊 Normalized Decision Matrix")
st.dataframe(pd.concat([alternative_names, normalized_matrix], axis=1))

# --- Apply Weights ---
weighted_matrix = normalized_matrix * weights
st.subheader("📌 Weighted Normalized Matrix")
st.dataframe(pd.concat([alternative_names, weighted_matrix], axis=1))

# --- Compute MOORA Scores ---
benefit_indices = [i for i, imp in enumerate(impacts) if imp == '+']
cost_indices = [i for i, imp in enumerate(impacts) if imp == '-']

moora_scores = weighted_matrix.iloc[:, benefit_indices].sum(axis=1) - weighted_matrix.iloc[:, cost_indices].sum(axis=1)

# --- Ranking ---
ranking = pd.DataFrame({
    "Alternative": alternative_names,
    "MOORA Score": moora_scores,
    "Rank": moora_scores.rank(ascending=False).astype(int)
}).sort_values(by="MOORA Score", ascending=False)

# --- Highlight Top Rank ---
st.subheader("🏆 Final Ranking (MOORA)")
def highlight_top(row):
    return ['background-color: lightgreen; font-weight: bold' if row['Rank'] == 1 else '' for _ in row]
st.dataframe(ranking.style.apply(highlight_top, axis=1), use_container_width=True)

# --- Visualization: Bar Chart of MOORA Scores ---
st.subheader("📈 MOORA Score Visualization")

st.bar_chart(ranking.set_index("Alternative")["MOORA Score"])

# --- Download Results ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='MOORA Results')
    return output.getvalue()

excel_data = to_excel(ranking)
st.download_button("📥 Download Results as Excel", data=excel_data, file_name="greenmoora_results.xlsx")


# --- Footer ---
st.markdown("---")
st.markdown("Created with 💚 for the theme *Celebrating Innovation, Commercialisation and Publication*. Powered by MOORA, Big Data & Streamlit.")
