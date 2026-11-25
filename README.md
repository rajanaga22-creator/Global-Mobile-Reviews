# Mobile Phone Recommendation System

This project is a simple **mobile phone recommendation web app** built with
Python and Streamlit. It uses mobile phone review data to suggest models that
are similar in price, customer rating, and features such as battery, camera,
performance, design, and display.

## What It Does

- Select a phone you already like.
- Set a maximum budget in USD.
- Set a minimum review rating.
- View a ranked list of similar phone recommendations.
- Explore phone groups created with clustering.
- Download the cleaned review dataset.

## How It Works

1. `Mobile Reviews Sentiment null.csv` contains the original review data.
2. `mob.py` cleans missing prices and ratings and creates one summary for each
   phone model.
3. Cosine similarity compares phone models and finds the closest matches.
4. The elbow method selects the K-means cluster count from within-cluster
   inertia, then groups similar phones. A silhouette score is shown separately
   as a clustering-quality metric.
5. `app.py` displays the recommendations and filters in a Streamlit website.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `mob.py` | Data cleaning, recommendation, and clustering logic |
| `mobile.ipynb` | Data analysis notebook |
| `Mobile Reviews Sentiment null.csv` | Original review dataset |
| `Mobile Reviews Sentiment cleaned.csv` | Processed dataset |
| `requirements.txt` | Required Python libraries |

## Installation

Use Python 3.10 or a later supported version. From the project folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the App

```powershell
python -m streamlit run app.py
```

Streamlit will open the app in your browser. You can also use the local URL
shown in the terminal.

## Prepare the Dataset

The app automatically cleans the source data when needed. To create the cleaned
CSV manually, run:

```powershell
python mob.py
```

This reads `Mobile Reviews Sentiment null.csv` and creates
`Mobile Reviews Sentiment cleaned.csv`. Missing values are filled for analysis,
while flags preserve whether a rating or sentiment value was originally
available.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
