import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import KBinsDiscretizer
from sentence_transformers import SentenceTransformer

# Load necessary models and tokenizer
token = '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'
tokenizer = AutoTokenizer.from_pretrained("gpt2", token=token)
llm_model = AutoModelForCausalLM.from_pretrained("gpt2", token=token)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def prompt_time_series(data, region, start_date, end_date):
    """
    Generates a natural language prompt from time series data.
    """
    daily_changes = np.diff(data)
    percent_changes = (daily_changes / data[:-1]) * 100  # Calculate daily % change
    prompt = f"From {start_date} to {end_date}, COVID-19 cases in {region} have changed as follows:\n"
    
    for i, change in enumerate(percent_changes):
        day = f"Day {i+1}"  # Or use actual date if available
        prompt += f"{day}: {'increased' if change > 0 else 'decreased'} by {abs(change):.2f}%.\n"

    prompt += "Based on these trends, forecast the cases for the next few days."
    return prompt

def quantize_time_series(data, n_bins=10):
    """
    Quantizes continuous time series data into discrete tokens for LLM processing.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    data_reshaped = data.reshape(-1, 1)
    quantized_data = discretizer.fit_transform(data_reshaped).flatten()
    tokens = ["bin_" + str(int(value)) for value in quantized_data]
    return tokens

def align_embeddings(time_series_data, context_data):
    """
    Aligns time series embeddings with LLM embeddings using a sentence transformer model.
    """
    # Generate embeddings for time series data
    time_series_str = " ".join([f"{int(value)}" for value in time_series_data])
    ts_embedding = sentence_model.encode(time_series_str)

    # Generate embeddings for context data
    context_embedding = sentence_model.encode(context_data)
    
    # Combine embeddings (can use concatenation or averaging)
    aligned_embedding = np.concatenate((ts_embedding, context_embedding))
    return aligned_embedding

def combine_with_gnn_embeddings(gnn_embeddings, llm_embeddings):
    """
    Combines GNN embeddings with LLM embeddings for enhanced feature representation.
    """
    # Ensure both embeddings are in tensor form for PyTorch compatibility
    gnn_embeddings_tensor = torch.tensor(gnn_embeddings)
    llm_embeddings_tensor = torch.tensor(llm_embeddings)

    # Check if llm_embeddings_tensor has only one dimension and add a new dimension if needed
    if llm_embeddings_tensor.dim() == 1:
        llm_embeddings_tensor = llm_embeddings_tensor.unsqueeze(0)  # Add batch dimension if needed

    # Repeat or expand llm_embeddings to match gnn_embeddings along batch dimension if necessary
    if gnn_embeddings_tensor.shape[0] != llm_embeddings_tensor.shape[0]:
        llm_embeddings_tensor = llm_embeddings_tensor.expand(gnn_embeddings_tensor.shape[0], -1)
    
    # Concatenate along the last dimension
    combined_embeddings = torch.cat((gnn_embeddings_tensor, llm_embeddings_tensor), dim=-1)
    return combined_embeddings

def run_llm_prediction(prompt):
    """
    Uses an LLM to generate a prediction based on a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=50)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# Main function to integrate with GNN
def integrate_gnn_with_llm(gnn_embeddings, data, region, start_date, end_date, context_data=None):
    """
    Full integration function for GNN and LLM with time series prompting, quantization, and alignment.
    """
    # Step 1: Create a prompt from time series data
    prompt = prompt_time_series(data, region, start_date, end_date)
    
    # Step 2: Quantize the time series data for LLM token compatibility
    quantized_data = quantize_time_series(data)
    
    # Step 3: Align embeddings if context data is available
    llm_embeddings = None
    if context_data:
        llm_embeddings = align_embeddings(data, context_data)
    
    # Step 4: Combine GNN and LLM embeddings
    if llm_embeddings is not None:
        combined_embeddings = combine_with_gnn_embeddings(gnn_embeddings, llm_embeddings)
    else:
        combined_embeddings = gnn_embeddings  # Use only GNN if no LLM data available
    
    # Step 5: Run LLM prediction based on the prompt
    prediction = run_llm_prediction(prompt)
    
    return {
        "prompt": prompt,
        "quantized_data": quantized_data,
        "combined_embeddings": combined_embeddings,
        "llm_prediction": prediction
    }
    
def evaluate_model(predictions, targets):
   
    # Convert predictions to a numpy array if necessary
        predictions = np.array(predictions)
        targets = np.array(targets)
    
    # Calculate evaluation metrics
        mae = np.mean(np.abs(predictions - targets))  # Mean Absolute Error
        mse = np.mean((predictions - targets) ** 2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100  # Mean Absolute Percentage Error

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# Example usage
if __name__ == "__main__":
    # Mock data
    time_series_data = np.array([100, 120, 130, 125, 135, 140, 150, 160, 155, 170])
    gnn_embeddings = np.random.rand(len(time_series_data), 64)  # Example GNN embeddings
    region = "Region X"
    start_date = "March 1"
    end_date = "March 10"
    context_data = "Social distancing policies were enacted during this period."

    results = integrate_gnn_with_llm(gnn_embeddings, time_series_data, region, start_date, end_date, context_data)
    print("Prompt:\n", results["prompt"])
    print("Quantized Data:\n", results["quantized_data"])
    print("Combined Embeddings Shape:\n", results["combined_embeddings"].shape)
    print("LLM Prediction:\n", results["llm_prediction"])
    true_values = np.array([102, 125, 128, 130, 145, 138, 148, 165, 158, 172])

    # Extract model predictions
    predictions = results["llm_prediction"]

    # Evaluate accuracy
    metrics = evaluate_model(predictions, true_values)
    print("Evaluation Metrics:", metrics)
