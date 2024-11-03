import requests
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import praw
from prawcore.exceptions import Redirect

# Replace with actual API keys
GNEWS_API_KEY = "35a6730b2d0f416c626b30fdcefdd616"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAP%2FxwgEAAAAAR993OB%2Fp%2F3pjcm7LOa5xLvqC%2BBc%3D3AmuGD8zBA5Ym6qTCHRxCrVH6LewGeKOIVag0bym6eiugOpa9V"
REDDIT_CLIENT_ID = "hF0Ws3F1sqI12JlOXj3ulw"
REDDIT_CLIENT_SECRET = "zshlI1tBgPVRNf3ZR54e0vnBmUzr7Q"

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="covid_analysis_app"
)

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

def fetch_reddit_data(subreddit_name, post_type="top", limit=30):
    """Fetch top or hot posts from a given subreddit."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        if post_type == "top":
            posts = subreddit.top(limit=limit)
        elif post_type == "hot":
            posts = subreddit.hot(limit=limit)
        else:
            posts = subreddit.new(limit=limit)

        # Extract the needed fields from each post
        return [
            {
                "title": post.title,
                "body": post.selftext,
                "upvotes": post.score,
                "num_comments": post.num_comments
            }
            for post in posts
        ]
    except Redirect:
        print(f"Warning: Subreddit '{subreddit_name}' does not exist or is restricted.")
        return []  # Return an empty list if subreddit is unavailable

def process_text_data(text_data, max_length=512):
    """Generate embeddings using a pre-trained language model, handling long inputs."""
    nlp_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = []

    for text in text_data:
        # Split text into chunks if it's longer than max_length tokens
        tokenized_text = nlp_model.tokenizer(text, return_tensors="pt", truncation=False)
        num_tokens = tokenized_text.input_ids.shape[1]

        if num_tokens > max_length:
            # Process text in chunks of max_length
            for i in range(0, num_tokens, max_length):
                chunk = text[i:i+max_length]
                chunk_embedding = nlp_model(chunk)[0][0]  # Get embedding for this chunk
                embeddings.append(chunk_embedding)
            # Compute the average embedding for all chunks
            avg_embedding = np.mean(embeddings, axis=0)
            embeddings.append(avg_embedding)
        else:
            # Process normally if text is within max_length
            embeddings.append(nlp_model(text)[0][0])

    return np.mean(embeddings, axis=0)  # Return the average embedding for stability


def generate_contextual_embeddings(region_names, start_date="2020-01-01", end_date="2021-12-31"):
    """Generate contextual embeddings for each region based on historical news, tweets, and Reddit posts."""
    contextual_embeddings = []

    # Define region-specific subreddits
    region_subreddits = {
    "US": "CoronavirusUS",
    "IT": "italy",
    "ES": "spain",
    "EN": "CoronavirusUK",
    "FR": "CoronavirusFR"
}


    for region in region_names:
        # Initialize an empty list for text data from all sources
        texts = []

        # Fetch news data
        news_data = fetch_contextual_data(GNEWS_API_KEY, "COVID-19", region, source='GNews', start_date=start_date, end_date=end_date)
        if news_data:
            texts.extend([item['title'] for item in news_data.get('articles', [])])

        # Fetch Twitter data
        tweet_data = fetch_contextual_data(TWITTER_BEARER_TOKEN, "COVID-19", region, source='Twitter', start_date=start_date, end_date=end_date)
        if tweet_data:
            texts.extend([tweet['text'] for tweet in tweet_data.get('data', [])])

        # Fetch Reddit data with error handling for missing subreddits
        subreddit_name = region_subreddits.get(region, "Coronavirus")
        try:
            reddit_posts = fetch_reddit_data(subreddit_name, post_type="top", limit=30)
        except Exception as e:
            print(f"Warning: Could not access subreddit '{subreddit_name}' for region '{region}'. Error: {e}")
            reddit_posts = []  # Set to an empty list if the subreddit is missing or restricted

        if reddit_posts:
            reddit_texts = [post.get('title', '') + ' ' + post.get('body', '') for post in reddit_posts]
            texts.extend(reddit_texts)

            # Gather engagement metrics, using defaults for missing data
            avg_upvotes = np.mean([post.get('upvotes', 0) for post in reddit_posts])
            avg_comments = np.mean([post.get('comments', 0) for post in reddit_posts])
        else:
            avg_upvotes, avg_comments = 0, 0

        # Generate embeddings for the combined text data
        if texts:
            region_embedding = process_text_data(texts)
            # Add engagement metrics to the feature vector if needed
            region_embedding = np.append(region_embedding, [avg_upvotes, avg_comments])
        else:
            region_embedding = np.zeros(384 + 2)  # Assuming 384-dim for text embedding + 2 for engagement metrics

        contextual_embeddings.append(region_embedding)

    # Scale embeddings and return
    scaler = MinMaxScaler()
    contextual_embeddings = scaler.fit_transform(contextual_embeddings)
    return contextual_embeddings


# # Example usage:
# region_names = ["US", "IT", "ES"]
# contextual_embeddings = generate_contextual_embeddings(region_names)
# print("Contextual Embeddings:\n", contextual_embeddings)
