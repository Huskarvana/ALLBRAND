import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import feedparser
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille Marques Auto", layout="wide")
st.title("🚗 Agent de Veille – Marques Concurrentes")

API_KEY_NEWSDATA = st.secrets["API_KEY_NEWSDATA"]
MEDIASTACK_API_KEY = st.secrets["MEDIASTACK_API_KEY"]

MARQUES = [
    "DS Automobiles", "Volvo", "BMW", "Audi", "Mercedes-Benz",
    "Peugeot", "Renault", "Citroën", "Lexus", "Jaguar", "Tesla"
]

LANGUES_DISPONIBLES = ["all", "fr", "en", "de", "es", "it", "pt", "nl"]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_analyzer = get_sentiment_pipeline()

def fetch_newsdata_articles(query, max_results=5, lang=None):
    params = {"apikey": API_KEY_NEWSDATA, "q": query}
    if lang and lang != "all":
        params["language"] = lang
    try:
        response = requests.get("https://newsdata.io/api/1/news", params=params)
        data = response.json()
        return [{
            "date": item.get("pubDate", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source_id", ""),
            "lien": item.get("link", "")
        } for item in data.get("results", [])[:max_results]]
    except:
        return []

def fetch_mediastack_articles(query, max_results=5, lang=None):
    params = {"access_key": MEDIASTACK_API_KEY, "keywords": query}
    if lang and lang != "all":
        params["languages"] = lang
    try:
        response = requests.get("http://api.mediastack.com/v1/news", params=params)
        data = response.json()
        return [{
            "date": item.get("published_at", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source", ""),
            "lien": item.get("url", "")
        } for item in data.get("data", [])[:max_results]]
    except:
        return []

def analyser_article(row):
    try:
        texte = row['contenu']
        if not texte or not isinstance(texte, str):
            raise ValueError("contenu vide")
        texte = texte[:512]
        prediction = sentiment_analyzer(texte)[0]
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        sentiment = label_map.get(prediction['label'], "Neutral")
    except:
        sentiment = "Neutral"
    résumé = row['contenu'][:200] + "..." if row['contenu'] else ""
    return pd.Series({'résumé': résumé, 'ton': sentiment})

# --- INTERFACE UTILISATEUR ---
st.sidebar.title("Filtres de veille")
filtre_marque = st.sidebar.selectbox("Choisir la marque à surveiller", MARQUES)
nb_articles = st.sidebar.slider("Nombre d'articles par source", 5, 30, 10)
filtre_langue = st.sidebar.selectbox("Filtrer par langue", LANGUES_DISPONIBLES)
filtre_ton = st.sidebar.selectbox("Filtrer par ton", ["Tous", "Positive", "Neutral", "Negative"])

if st.button("🔍 Lancer la veille"):
    lang = None if filtre_langue == "all" else filtre_langue
    newsdata = fetch_newsdata_articles(filtre_marque, nb_articles, lang)
    mediastack = fetch_mediastack_articles(filtre_marque, nb_articles, lang)

    articles = pd.DataFrame(newsdata + mediastack)

    if not articles.empty:
        with st.spinner("Analyse des articles..."):
            articles[['résumé', 'ton']] = articles.apply(analyser_article, axis=1)

        articles['date'] = pd.to_datetime(articles['date'], errors='coerce')
        articles = articles.sort_values(by='date', ascending=False)

        if filtre_ton != "Tous":
            articles = articles[articles['ton'] == filtre_ton]

        st.dataframe(articles[['date', 'titre', 'ton', 'résumé', 'source', 'lien']])
    else:
        st.warning("Aucun article trouvé.")