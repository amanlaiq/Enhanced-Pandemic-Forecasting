import requests
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

# Replace with actual API keys
GNEWS_API_KEY = "35a6730b2d0f416c626b30fdcefdd616"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAP%2FxwgEAAAAAR993OB%2Fp%2F3pjcm7LOa5xLvqC%2BBc%3D3AmuGD8zBA5Ym6qTCHRxCrVH6LewGeKOIVag0bym6eiugOpa9V"

def fetch_contextual_data(api_key, query, region, source='GNews'):
    """Fetch recent news or tweets about COVID-19 for a given region."""
    if source == 'GNews':
        url = f'https://gnews.io/api/v4/search?q={query}&lang=en&country={region}&token={api_key}'
        headers = {}
    elif source == 'Twitter':
        url = f'https://api.twitter.com/2/tweets/search/recent?query={query}%20place_country:{region}&tweet.fields=created_at&expansions=geo.place_id&user.fields=location'
        headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None

def process_text_data(text_data):
    """Generate embeddings using a pre-trained language model."""
    nlp_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [nlp_model(text)[0][0] for text in text_data]  # Get the first embedding from each text
    return np.mean(embeddings, axis=0)  # Return the average embedding for stability

def generate_contextual_embeddings(region_names):
    """Generate contextual embeddings for each region based on news and tweets."""
    contextual_embeddings = []

    for region in region_names:
        # Fetch news and tweets
        news_data = fetch_contextual_data(GNEWS_API_KEY, "COVID-19", region, source='GNews')
        tweet_data = fetch_contextual_data(TWITTER_BEARER_TOKEN, "COVID-19", region, source='Twitter')
        
        # Collect text snippets from fetched data
        texts = []
        if news_data:
            texts.extend([item['title'] for item in news_data.get('articles', [])])
        if tweet_data:
            texts.extend([tweet['text'] for tweet in tweet_data.get('data', [])])
        
        # Generate embeddings for the texts if any text data is available
        if texts:
            region_embedding = process_text_data(texts)
        else:
            region_embedding = np.zeros(384)  # Assuming 384-dim for default if no text is available

        contextual_embeddings.append(region_embedding)

    # Scale embeddings and return
    scaler = MinMaxScaler()
    contextual_embeddings = scaler.fit_transform(contextual_embeddings)
    return contextual_embeddings
