import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set the device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and Preprocess Data
def load_data(case_file):
    case_data = pd.read_csv(case_file, index_col=0)
    time_series_data = case_data.iloc[:, 1:].values
    return time_series_data

def quantize_time_series(data, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    data_reshaped = data.reshape(-1, 1)
    quantized_data = discretizer.fit_transform(data_reshaped).flatten()
    return quantized_data, discretizer

def create_sequences(data, sequence_length=30):
    inputs, targets = [], []
    for i in range(len(data) - sequence_length):
        inputs.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(inputs), np.array(targets)

# Step 2: Define Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets), "Inputs and targets must have the same length"
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.targets[idx], dtype=torch.long),
        }

# Step 3: Fine-Tune GPT4TS
def fine_tune_gpt4ts(X_train, y_train, X_val, y_val, vocab_size):
    """
    Fine-tune GPT4TS (GPT-2) on the time-series dataset.
    """
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Assign PAD token to EOS token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(vocab_size)

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Masked language modeling is disabled for GPT
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()
    return model

# Step 4: Generate Predictions
def generate_predictions(model, input_sequence, num_predictions):
    model.eval()
    input_ids = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(num_predictions):
            outputs = model.generate(input_ids, max_new_tokens=1)
            next_token = outputs[0, -1].item()
            predictions.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)

    return predictions

# Step 5: Evaluate Model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# Step 6: Main Function
def main():
    case_file = "/Users/jagritibhandari/Desktop/Pandemic-Forecasting/data/Italy/italy_labels.csv"
    time_series_data = load_data(case_file)
    quantized_data, discretizer = quantize_time_series(time_series_data.flatten(), n_bins=50)

    sequence_length = 30
    X, y = create_sequences(quantized_data, sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    vocab_size = len(set(quantized_data))
    model = fine_tune_gpt4ts(X_train, y_train, X_val, y_val, vocab_size)

    last_sequence = X_val[-1]
    num_predictions = 10
    predicted_tokens = generate_predictions(model, last_sequence, num_predictions)

    predicted_values = discretizer.inverse_transform(np.array(predicted_tokens).reshape(-1, 1))
    print("Predicted values for the next 10 steps:", predicted_values.flatten())

    y_val_decoded = discretizer.inverse_transform(y_val.reshape(-1, 1))
    mse, mae = evaluate_model(y_val_decoded[:num_predictions], predicted_values)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()
