import requests
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

# Replace with actual API keys
GNEWS_API_KEY = "35a6730b2d0f416c626b30fdcefdd616"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAP%2FxwgEAAAAAR993OB%2Fp%2F3pjcm7LOa5xLvqC%2BBc%3D3AmuGD8zBA5Ym6qTCHRxCrVH6LewGeKOIVag0bym6eiugOpa9V"

def fetch_contextual_data(api_key, query, region, source='GNews', start_date=None, end_date=None):
    """Fetch historical news or tweets about COVID-19 for a given region and date range."""
    if source == 'GNews':
        url = f'https://gnews.io/api/v4/search?q={query}&lang=en&country={region}&token={api_key}'
        if start_date and end_date:
            url += f"&from={start_date}&to={end_date}"
        headers = {}
    elif source == 'Twitter':
        url = f'https://api.twitter.com/2/tweets/search/recent?query={query}%20place_country:{region}'
        if start_date:
            url += f"&start_time={start_date}T00:00:00Z"  # start_time in ISO format
        if end_date:
            url += f"&end_time={end_date}T00:00:00Z"  # end_time in ISO format
        headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None

def process_text_data(text_data):
    """Generate embeddings using a pre-trained language model."""
    nlp_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [nlp_model(text)[0][0] for text in text_data]  # Get the first embedding from each text
    return np.mean(embeddings, axis=0)  # Return the average embedding for stability

def generate_contextual_embeddings(region_names, start_date="2020-01-01", end_date="2021-12-31"):
    """Generate contextual embeddings for each region based on historical news and tweets."""
    contextual_embeddings = []

    for region in region_names:
        # Fetch news and tweets from the specified date range
        news_data = fetch_contextual_data(GNEWS_API_KEY, "COVID-19", region, source='GNews', start_date=start_date, end_date=end_date)
        tweet_data = fetch_contextual_data(TWITTER_BEARER_TOKEN, "COVID-19", region, source='Twitter', start_date=start_date, end_date=end_date)
        
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

# # Example usage:
# region_names = ["US", "IT", "ES"]
# contextual_embeddings = generate_contextual_embeddings(region_names)
# print("Contextual Embeddings:\n", contextual_embeddings)
