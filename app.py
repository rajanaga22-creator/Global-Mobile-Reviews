"""Streamlit interface for the mobile phone recommendation demo."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from mob import (
    DEFAULT_OUTPUT,
    build_product_catalog,
    clean_mobile_data,
    cluster_products,
    recommend_similar_phones,
)


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT_DIR / "Mobile Reviews Sentiment null.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    if DEFAULT_OUTPUT.exists():
        return pd.read_csv(DEFAULT_OUTPUT)
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Could not find source dataset: {SOURCE_CSV}")
    cleaned = clean_mobile_data(pd.read_csv(SOURCE_CSV))
    cleaned.to_csv(DEFAULT_OUTPUT, index=False)
    return cleaned


st.set_page_config(page_title="Mobile Phone Recommender", page_icon="📱", layout="wide")
st.title("Mobile Phone Recommendation System")
st.write(
    "Choose a phone you like and discover similar models based on price, "
    "observed review ratings, and feature scores."
)

try:
    mobile_data = load_data()
    product_catalog = build_product_catalog(mobile_data)
except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load recommendation data: {error}")
    st.stop()

product_catalog["product_label"] = (
    product_catalog["brand"] + " " + product_catalog["model"]
)
if product_catalog.empty:
    st.error("No phone models have enough review data to make recommendations.")
    st.stop()

clustered_products = cluster_products(
    product_catalog, n_clusters=min(4, len(product_catalog))
)

with st.sidebar:
    st.header("Recommendation preferences")
    selected_product = st.selectbox(
        "A phone you like",
        options=product_catalog["product_label"].sort_values().tolist(),
    )
    max_price = max(1, math.ceil(product_catalog["avg_price_usd"].max() / 50) * 50)
    budget = st.slider(
        "Maximum average price (USD)",
        min_value=0,
        max_value=max_price,
        value=max_price,
        step=50,
    )
    min_rating = st.slider(
        "Minimum average review rating",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
    )
    result_count = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=min(10, max(1, len(product_catalog) - 1)),
        value=min(5, max(1, len(product_catalog) - 1)),
        step=1,
    )

selected_details = product_catalog.loc[
    product_catalog["product_label"].eq(selected_product)
].iloc[0]
detail_columns = st.columns(3)
detail_columns[0].metric("Average price", f"${selected_details['avg_price_usd']:,.0f}")
detail_columns[1].metric(
    "Observed average rating",
    (
        f"{selected_details['avg_rating']:.2f} / 5"
        if pd.notna(selected_details["avg_rating"])
        else "Not available"
    ),
)
detail_columns[2].metric("Reviews", f"{int(selected_details['review_count']):,}")

recommendations = recommend_similar_phones(
    product_catalog,
    selected_product,
    max_price_usd=float(budget),
    min_rating=float(min_rating),
    limit=int(result_count),
)

st.subheader("Recommended phones")
if recommendations.empty:
    st.info("No phones match these filters. Try increasing your budget or lowering the minimum rating.")
else:
    shown_recommendations = recommendations.rename(
        columns={
            "product_label": "Phone",
            "avg_price_usd": "Average price (USD)",
            "avg_rating": "Average rating",
            "review_count": "Reviews",
            "similarity": "Similarity",
        }
    )
    st.dataframe(
        shown_recommendations[
            [
                "Phone",
                "Average price (USD)",
                "Average rating",
                "Reviews",
                "Similarity",
            ]
        ].round(2),
        hide_index=True,
        width="stretch",
    )

st.caption(
    "Recommendations use item attributes and cosine similarity; they are a "
    "demonstration, not a guarantee of personal preference. Ratings are averaged "
    "from observed reviews only."
)

st.subheader("Phone clusters")
st.write(
    "Phone models are grouped by review count, price, observed rating, and "
    "feature ratings. Cluster numbers are labels, not rankings."
)
for segment, group in clustered_products.groupby("product_segment", sort=True):
    st.markdown(f"### Cluster {segment + 1}")
    metrics = st.columns(3)
    metrics[0].metric("Phone models", f"{len(group):,}")
    metrics[1].metric("Average price", f"${group['avg_price_usd'].mean():,.0f}")
    metrics[2].metric("Average rating", f"{group['avg_rating'].mean():.2f} / 5")

    cluster_display = group.rename(
        columns={
            "brand": "Brand",
            "model": "Model",
            "avg_price_usd": "Average price (USD)",
            "avg_rating": "Average rating",
            "review_count": "Reviews",
        }
    )
    st.dataframe(
        cluster_display[
            ["Brand", "Model", "Average price (USD)", "Average rating", "Reviews"]
        ]
        .sort_values(["Brand", "Model"])
        .round(2),
        hide_index=True,
        width="stretch",
    )

with st.expander("About the data"):
    st.write(
        f"The cleaned dataset contains {len(mobile_data):,} reviews and "
        f"{len(product_catalog):,} phone models. Missing rating and price values "
        "are filled for analysis, with flags retained in the cleaned CSV so "
        "imputed ratings are not counted as customer feedback."
    )
    st.download_button(
        "Download cleaned CSV",
        data=mobile_data.to_csv(index=False).encode("utf-8"),
        file_name=DEFAULT_OUTPUT.name,
        mime="text/csv",
    )
