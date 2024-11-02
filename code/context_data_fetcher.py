# context_data_fetcher.py

import requests
import tweepy
from transformers import AutoTokenizer, AutoModel
import torch

# Set up API keys
GNEWS_API_KEY = '35a6730b2d0f416c626b30fdcefdd616'
TWITTER_API_KEY = 'T7NzGEV9wkce3EjtIlJBSvZ9T'

# Set up Twitter API with Tweepy
auth = tweepy.AppAuthHandler(TWITTER_API_KEY, '')
twitter_api = tweepy.API(auth)

# Load LLM model and tokenizer for embeddings
model_name = 'bert-base-uncased'  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModel.from_pretrained(model_name)
llm_model.eval()

def fetch_news(region, query="COVID-19"):
    """
    Fetch recent news articles from GNews API related to COVID-19 for the specified region.
    """
    url = f"https://gnews.io/api/v4/search?q={query}+{region}&token={GNEWS_API_KEY}&lang=en&max=5"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [article['title'] for article in articles]
    else:
        print("Failed to fetch news:", response.status_code)
        return []

def fetch_twitter_sentiment(region, query="COVID-19"):
    """
    Fetch recent tweets related to COVID-19 and region and generate sentiment-based embeddings.
    """
    tweets = twitter_api.search(q=f"{query} {region}", lang="en", count=5)
    tweet_texts = [tweet.text for tweet in tweets]
    return tweet_texts

def get_text_embedding(text):
    """
    Convert text into an embedding using LLM.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = llm_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0)

def generate_contextual_embeddings(region_names):
    """
    Generate contextual embeddings for each region by combining news and sentiment data.
    """
    region_embeddings = {}
    for region in region_names:
        news_data = fetch_news(region)
        sentiment_data = fetch_twitter_sentiment(region)
        combined_text = " ".join(news_data + sentiment_data)
        embedding = get_text_embedding(combined_text)
        region_embeddings[region] = embedding
    return region_embeddings
