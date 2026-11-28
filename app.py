"""Streamlit interface for the mobile phone recommendation demo."""

from __future__ import annotations

import math
from pathlib import Path
import json
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st

from mob import (
    DEFAULT_OUTPUT,
    build_product_catalog,
    calculate_product_silhouette,
    clean_mobile_data,
    cluster_products,
    recommend_similar_phones,
    tune_product_clustering,
)


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT_DIR / "Mobile Reviews Sentiment null.csv"


@st.cache_data(ttl=24 * 60 * 60)
def get_phone_image_url(brand: str, model: str) -> str | None:
    """Look up a thumbnail from Wikipedia for a brand and model."""
    query = urllib.parse.quote(f"{brand} {model}")
    url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query&generator=search&gsrlimit=1&prop=pageimages"
        f"&gsrsearch={query}&piprop=thumbnail&pithumbsize=500&pilimit=1&format=json"
    )
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Global-Mobile-Reviews/1.0"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            data = json.load(response)
    except (OSError, json.JSONDecodeError):
        return None

    for page in data.get("query", {}).get("pages", {}).values():
        thumbnail = page.get("thumbnail", {}).get("source")
        if thumbnail:
            return thumbnail
    return None


def render_phone_card(phone: pd.Series) -> None:
    """Render a product card with an optional Wikimedia thumbnail."""
    brand = str(phone["brand"])
    model = str(phone["model"])
    image_url = phone.get("image_url") if "image_url" in phone else None
    if not isinstance(image_url, str) or not image_url.strip():
        image_url = get_phone_image_url(brand, model)

    if image_url:
        st.image(image_url, use_column_width=True)
    else:
        st.markdown(
            "<div style='height:220px;display:flex;align-items:center;"
            "justify-content:center;border:1px solid #ddd;border-radius:8px;"
            "font-size:64px'>📱</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"**{brand} {model}**")
    st.write(f"${phone['avg_price_usd']:,.0f}")
    rating = (
        f"{phone['avg_rating']:.2f} / 5"
        if pd.notna(phone["avg_rating"])
        else "Rating unavailable"
    )
    st.caption(f"{rating} · {int(phone['review_count']):,} reviews")
    if "similarity" in phone:
        st.caption(f"Similarity: {phone['similarity']:.0%}")


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

product_tuning = tune_product_clustering(product_catalog)
product_cluster_count = int(product_tuning["n_clusters"])
product_feature_columns = list(product_tuning["feature_columns"])
clustered_products = cluster_products(
    product_catalog,
    n_clusters=product_cluster_count,
    feature_columns=product_feature_columns,
)
product_silhouette = (
    calculate_product_silhouette(
        product_catalog,
        n_clusters=product_cluster_count,
        feature_columns=product_feature_columns,
    )
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
    for start in range(0, len(recommendations), 5):
        cards = recommendations.iloc[start:start + 5]
        card_columns = st.columns(len(cards))
        for column, (_, phone) in zip(card_columns, cards.iterrows()):
            with column:
                with st.container(border=True):
                    render_phone_card(phone)

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
        use_container_width=True,
    )

st.caption(
    "Recommendations use item attributes and cosine similarity; they are a "
    "demonstration, not a guarantee of personal preference. Ratings are averaged "
    "from observed reviews only."
)

st.subheader("Phone clusters")
st.write(
    "The elbow method selects the cluster count from the within-cluster "
    "inertia curve. Phone models are grouped by review count, price, observed "
    "rating, and feature ratings. Cluster numbers are labels, not rankings."
)
if product_silhouette is not None:
    st.metric("Silhouette score", f"{product_silhouette:.3f}")
    st.caption(
        "Silhouette scores range from -1 to 1. Higher values indicate better "
        "separated and more cohesive clusters; this score does not select "
        "the cluster count."
    )
else:
    st.metric("Silhouette score", "Not available")
st.write(
    f"Elbow-selected configuration: **{product_cluster_count} clusters** "
    f"using {len(product_feature_columns)} tuned features."
)
inertia_curve = pd.DataFrame(product_tuning["inertia_curve"])
if not inertia_curve.empty:
    st.line_chart(inertia_curve.set_index("n_clusters"))
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
        use_container_width=True,
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
