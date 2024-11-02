# In a new module file, e.g., llm_integration.py, or in metalearn.py directly if you prefer:

import numpy as np

def generate_prompt(node_embeddings, predictions, historical_data, country_name):
    """
    Generates a structured prompt for LLM interpretation.
    
    Parameters:
    - node_embeddings (np.array): GNN-generated node embeddings.
    - predictions (np.array): Predictions from the GNN model.
    - historical_data (np.array): Historical pandemic data.
    - country_name (str): Name of the country for context in the prompt.
    
    Returns:
    - prompt (str): Structured prompt for the LLM to interpret.
    """
    # Summarize the historical data into recent trends
    recent_trend = historical_data[-7:].mean()  # Average of the last week as an example
    current_prediction = predictions[-1]  # Latest prediction for the upcoming period
    
    # Node embeddings summary (mean of embeddings for simplicity)
    embedding_summary = node_embeddings.mean(axis=0)

    # Generate the prompt string
    prompt = (
        f"Forecasting report for {country_name}:\n\n"
        f"Recent COVID-19 trend:\n"
        f" - The average cases over the past week are approximately {recent_trend:.2f}.\n"
        f" - Predicted cases for the next period are {current_prediction:.2f}.\n\n"
        f"GNN-encoded location embeddings summarize key region characteristics.\n"
        f"The embedding-based summary is as follows:\n"
        f" - Embedding vector: {embedding_summary.tolist()}.\n"
        f"Use this information to interpret potential COVID-19 trajectories."
    )
    
    return prompt



def quantize_data(predictions, bins=5):
    """
    Quantizes GNN prediction data into discrete tokens for LLM compatibility.
    
    Parameters:
    - predictions (np.array): Array of prediction values.
    - bins (int): Number of quantization levels or categories.
    
    Returns:
    - tokenized_data (list): List of quantized tokens representing the data.
    """
    # Define bin edges based on the range of predictions
    min_val, max_val = predictions.min(), predictions.max()
    bins = np.linspace(min_val, max_val, bins + 1)  # Create equal-width bins
    
    # Digitize data based on these bins (discretize into tokens)
    tokenized_data = np.digitize(predictions, bins)  # Returns bin indices
    
    return tokenized_data.tolist()

# Example usage:
# Assuming `predictions` is a np.array of GNN-generated forecast values.
# tokens = quantize_data(predictions, bins=5)
# print("Quantized tokens:", tokens)

# Example usage:
# Assuming `node_embeddings`, `predictions`, and `historical_data` are np.arrays available from GNN outputs.
# prompt_text = generate_prompt(node_embeddings, predictions, historical_data, "Italy")
# print(prompt_text)
