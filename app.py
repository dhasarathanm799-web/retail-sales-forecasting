import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #f0f2f6;
    padding: 18px;
    border-radius: 12px;
}

h1 {
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("submission.csv", encoding="utf-8-sig")
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    if "sales" not in data.columns or "id" not in data.columns:
        st.error("CSV must contain 'id' and 'sales' columns.")
        st.stop()

    return data

data = load_data()

# ---------------- HEADER ----------------
st.title("📊 Retail Sales Forecasting")
st.caption("Machine Learning | XGBoost | Interactive Analytics")

st.markdown("""
This dashboard presents predicted retail sales from a machine learning
forecasting project.
""")

st.divider()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Dashboard Filters")

min_id = int(data["id"].min())
max_id = int(data["id"].max())

selected_ids = st.sidebar.slider(
    "Select ID range",
    min_value=min_id,
    max_value=max_id,
    value=(min_id, max_id)
)

min_sales = float(data["sales"].min())
max_sales = float(data["sales"].max())

selected_sales = st.sidebar.slider(
    "Select predicted sales range",
    min_value=min_sales,
    max_value=max_sales,
    value=(min_sales, max_sales)
)

filtered_data = data[
    (data["id"] >= selected_ids[0]) &
    (data["id"] <= selected_ids[1]) &
    (data["sales"] >= selected_sales[0]) &
    (data["sales"] <= selected_sales[1])
]

# ---------------- SUMMARY METRICS ----------------
st.subheader("📌 Prediction Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    f"{len(filtered_data):,}"
)

col2.metric(
    "Average Predicted Sales",
    f"{filtered_data['sales'].mean():,.2f}"
    if not filtered_data.empty else "0"
)

col3.metric(
    "Total Predicted Sales",
    f"{filtered_data['sales'].sum():,.2f}"
)

col4.metric(
    "Highest Predicted Sales",
    f"{filtered_data['sales'].max():,.2f}"
    if not filtered_data.empty else "0"
)

st.divider()

# ---------------- INTERACTIVE CHARTS ----------------
st.subheader("📈 Interactive Sales Analysis")

if filtered_data.empty:
    st.warning("No predictions match the selected filters.")
else:
    chart_tab1, chart_tab2 = st.tabs([
        "Sales Trend",
        "Sales Distribution"
    ])

    with chart_tab1:
        chart_data = filtered_data.sort_values("id")

        fig = px.line(
            chart_data,
            x="id",
            y="sales",
            title="Predicted Sales by ID",
            labels={
                "id": "Prediction ID",
                "sales": "Predicted Sales"
            }
        )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_tab2:
        fig = px.histogram(
            filtered_data,
            x="sales",
            nbins=40,
            title="Distribution of Predicted Sales",
            labels={"sales": "Predicted Sales"}
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- DATA TABLE ----------------
st.subheader("📋 Prediction Records")

st.dataframe(
    filtered_data,
    use_container_width=True,
    height=350
)

# ---------------- DOWNLOAD ----------------
st.download_button(
    label="⬇️ Download Filtered Predictions",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_retail_sales_predictions.csv",
    mime="text/csv"
)

st.divider()

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("🤖 Model Performance")

st.info("""
The project uses machine learning models for retail sales forecasting.

The baseline Random Forest model previously achieved:

- MAE: 153.59
- RMSE: 450.53
- R² Score: 0.8936

These are preliminary baseline results, not verified performance
metrics for the tuned XGBoost model.
""")

# ---------------- PROJECT FOOTER ----------------
st.markdown("""
---
**Retail Sales Forecasting Project**  
Built with Python, Pandas, XGBoost, and Streamlit.
""")