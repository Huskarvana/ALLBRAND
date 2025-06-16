
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille Concurrence – Marques Auto", layout="wide")
st.title("🚗 Agent de Veille – Concurrence Marques SUV")

API_KEY_NEWSDATA = st.secrets["API_KEY_NEWSDATA"]

# Marques concurrentes (ex: SUV premium)
CONCURRENTES = [
    "Volvo", "Audi", "BMW", "Mercedes", "Peugeot", "Renault", "Citroën", 
    "Volkswagen", "Skoda", "Jeep", "Toyota", "Hyundai", "Kia"
]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_analyzer = get_sentiment_pipeline()

def fetch_articles_newsdata(query, lang="fr", max_results=10):
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": API_KEY_NEWSDATA,
        "q": query,
        "language": lang,
        "page": 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "date": item.get("pubDate", ""),
                "titre": item.get("title", ""),
                "contenu": item.get("description", ""),
                "source": item.get("source_id", ""),
                "lien": item.get("link", ""),
                "langue": item.get("language", "")
            })
        return results
    except Exception as e:
        st.error(f"Erreur API NewsData: {e}")
        return []

def analyser_article(row):
    try:
        sentiment = sentiment_analyzer(row['contenu'][:512])[0]['label']
    except:
        sentiment = "neutral"
    résumé = row['contenu'][:200] + "..." if row['contenu'] else ""
    return pd.Series({'résumé': résumé, 'ton': sentiment.capitalize()})

# --- UI ---
marque = st.selectbox("Choisir une marque concurrente", CONCURRENTES)
langue = st.selectbox("Langue", ["fr", "en", "es"])
if st.button("🔍 Lancer la veille"):
    articles = fetch_articles_newsdata(marque, lang=langue)
    if articles:
        df = pd.DataFrame(articles)
        df[['résumé', 'ton']] = df.apply(analyser_article, axis=1)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by='date', ascending=False)
        st.dataframe(df[['date', 'titre', 'ton', 'résumé', 'source', 'langue', 'lien']])
    else:
        st.warning("Aucun article trouvé.")
