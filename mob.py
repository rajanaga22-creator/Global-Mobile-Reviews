"""Clean mobile review data and provide phone-similarity recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "Mobile Reviews Sentiment null.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "Mobile Reviews Sentiment cleaned.csv"

REQUIRED_COLUMNS = {
    "brand",
    "model",
    "price_usd",
    "price_local",
    "currency",
    "exchange_rate_to_usd",
    "rating",
    "sentiment",
    "review_date",
    "source",
    "customer_name",
    "age",
    "verified_purchase",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "helpful_votes",
}

PRODUCT_FEATURES = [
    "avg_price_usd",
    "avg_rating",
    "battery_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
]
PRODUCT_CLUSTER_FEATURES = ["review_count", *PRODUCT_FEATURES]


def clean_mobile_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of mobile review data, retaining imputation flags."""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The input data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned = data.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].str.strip().replace("", pd.NA)

    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned["rating_was_observed"] = cleaned["rating"].notna()
    cleaned["sentiment_was_observed"] = cleaned["sentiment"].notna()

    cleaned["price_usd"] = pd.to_numeric(cleaned["price_usd"], errors="coerce")
    usd_by_model = cleaned.groupby("model")["price_usd"].transform("median")
    usd_global_median = cleaned["price_usd"].median()
    if pd.isna(usd_global_median):
        raise ValueError("Cannot fill price_usd because every value is missing.")
    cleaned["price_usd"] = cleaned["price_usd"].fillna(usd_by_model).fillna(
        usd_global_median
    )

    cleaned["exchange_rate_to_usd"] = pd.to_numeric(
        cleaned["exchange_rate_to_usd"], errors="coerce"
    )
    exchange_rate_by_currency = cleaned.groupby("currency")[
        "exchange_rate_to_usd"
    ].transform("median")
    cleaned["exchange_rate_to_usd"] = cleaned["exchange_rate_to_usd"].fillna(
        exchange_rate_by_currency
    )
    if cleaned["exchange_rate_to_usd"].isna().any():
        raise ValueError("Cannot convert local prices: exchange rates are missing.")

    cleaned["price_local_original"] = cleaned["price_local"].fillna("Missing")
    local_price_text = (
        cleaned["price_local"]
        .astype("string")
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    cleaned["price_local"] = pd.to_numeric(local_price_text, errors="coerce")
    converted_usd_price = cleaned["price_usd"] * cleaned["exchange_rate_to_usd"]
    cleaned["price_local"] = cleaned["price_local"].fillna(converted_usd_price)
    local_median_by_currency = cleaned.groupby("currency")["price_local"].transform(
        "median"
    )
    cleaned["price_local"] = cleaned["price_local"].fillna(local_median_by_currency)
    if cleaned["price_local"].isna().any():
        raise ValueError("Some local prices could not be filled.")

    rating_by_model = cleaned.groupby("model")["rating"].transform("median")
    rating_global_median = cleaned["rating"].median()
    if pd.isna(rating_global_median):
        raise ValueError("Cannot fill rating because every value is missing.")
    cleaned["rating"] = (
        cleaned["rating"]
        .fillna(rating_by_model)
        .fillna(rating_global_median)
        .round()
        .astype("int64")
    )

    observed_rating_sentiment = {
        1: "Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Positive",
    }
    sentiment_from_observed_rating = data["rating"].map(observed_rating_sentiment)
    cleaned["sentiment"] = cleaned["sentiment"].fillna(
        sentiment_from_observed_rating
    )
    cleaned["sentiment"] = cleaned["sentiment"].fillna("Unknown")
    cleaned["sentiment"] = cleaned["sentiment"].replace(
        {
            "positive": "Positive",
            "neutral": "Neutral",
            "negative": "Negative",
            "unknown": "Unknown",
        }
    )

    cleaned["source"] = cleaned["source"].fillna("Unknown")
    cleaned["review_date"] = pd.to_datetime(
        cleaned["review_date"], format="mixed", errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    numeric_columns = [
        "age",
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating",
        "helpful_votes",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def load_cleaned_data(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load cleaned data or generate it from the project's original CSV."""
    if path.exists():
        return pd.read_csv(path)
    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}, and source data not found at "
            f"{DEFAULT_INPUT}."
        )
    cleaned = clean_mobile_data(pd.read_csv(DEFAULT_INPUT))
    cleaned.to_csv(path, index=False)
    return cleaned


def build_product_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review rows into one row per phone model."""
    required = {"brand", "model", "price_usd", "rating", "rating_was_observed"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Cleaned data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    observed = data.copy()
    observed["observed_rating"] = observed["rating"].where(
        observed["rating_was_observed"].astype(bool)
    )
    feature_columns = [
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating",
    ]
    return (
        observed.groupby(["brand", "model"], as_index=False)
        .agg(
            review_count=("model", "size"),
            avg_price_usd=("price_usd", "mean"),
            avg_rating=("observed_rating", "mean"),
            battery_rating=("battery_life_rating", "mean"),
            camera_rating=("camera_rating", "mean"),
            performance_rating=("performance_rating", "mean"),
            design_rating=("design_rating", "mean"),
            display_rating=("display_rating", "mean"),
        )
        .dropna(subset=PRODUCT_FEATURES)
    )


def cluster_products(
    catalog: pd.DataFrame, n_clusters: int = 4
) -> pd.DataFrame:
    """Assign each phone model to a cluster based on its reviews and features."""
    missing_columns = set(PRODUCT_CLUSTER_FEATURES).difference(catalog.columns)
    if missing_columns:
        raise ValueError(
            "Product catalog is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if not 1 <= n_clusters <= len(catalog):
        raise ValueError(
            "n_clusters must be between 1 and the number of phone models."
        )

    clustered = catalog.copy()
    clusterer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    )
    clustered["product_segment"] = clusterer.fit_predict(
        clustered[PRODUCT_CLUSTER_FEATURES]
    )
    return clustered


def recommend_similar_phones(
    catalog: pd.DataFrame,
    selected_product: str,
    *,
    max_price_usd: float | None = None,
    min_rating: float = 0.0,
    limit: int = 5,
) -> pd.DataFrame:
    """Rank phones similar to a selected phone using standardized attributes."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if selected_product not in catalog["product_label"].values:
        raise ValueError(f"Unknown phone: {selected_product}")

    features = catalog[PRODUCT_FEATURES]
    scaled_features = StandardScaler().fit_transform(features)
    similarity = cosine_similarity(scaled_features)
    selected_index = catalog.index[
        catalog["product_label"].eq(selected_product)
    ][0]
    selected_position = catalog.index.get_loc(selected_index)

    results = catalog.copy()
    results["similarity"] = similarity[selected_position]
    results = results.loc[results["product_label"].ne(selected_product)]
    results = results.loc[results["avg_rating"].ge(min_rating)]
    if max_price_usd is not None:
        results = results.loc[results["avg_price_usd"].le(max_price_usd)]

    return results.sort_values(
        ["similarity", "avg_rating"], ascending=False
    ).head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean mobile review data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_data = pd.read_csv(args.input)
    cleaned_data = clean_mobile_data(raw_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(args.output, index=False)

    print(f"Saved cleaned data to: {args.output}")
    print(f"Rows: {len(cleaned_data):,}")
    print(f"Originally missing ratings: {(~cleaned_data['rating_was_observed']).sum():,}")
    print(f"Missing values remaining: {int(cleaned_data.isna().sum().sum()):,}")


if __name__ == "__main__":
    main()
