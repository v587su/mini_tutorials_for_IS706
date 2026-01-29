# A demo for training a sentiment analysis model using Hugging Face Transformers
# Prerequisites: 
# 1. Install Huggingface Transformers and Datasets: pip install transformers datasets
# 2. Install PyTorch (CPU-only version): pip install torch --index-url https://download.pytorch.org/whl/cpu
# 3. Install Numpy: pip install numpy


# ----------------------
# 1. Import Libraries
# ----------------------
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
import numpy as np

# ----------------------
# 2. Load Tiny Dataset
# ----------------------
# Load the IMDB dataset from Huggingface: comment-label pairs
dataset = load_dataset("imdb", split="train[:500]").train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ----------------------
# 3. Tokenize Text
# ----------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Format for PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------
# 4. Evaluation Metrics
# ----------------------
def compute_metrics(eval_pred):
    """Manual calculation of accuracy"""
    logits, labels = eval_pred
    
    # Step 1: Convert logits to predictions (0 or 1)
    predictions = np.argmax(logits, axis=1)
    
    # Step 2: Calculate accuracy (correct predictions / total predictions)
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = correct / total
    
    return {
        "accuracy": accuracy, 
    }

# ----------------------
# 5. Load Model
# ----------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    num_labels=2  # Negative (0) / Positive (1)
)

# ----------------------
# 6. Training Arguments
# ----------------------
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=1,              # Few epochs (fast training)
    per_device_train_batch_size=8,   # Small batch size for CPU
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",  # Added to see progress during training
    use_cpu=True,                 # Modern way to force CPU
    fp16=False
)

# ----------------------
# 7. Train the Model
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics  # Use our simplified metrics
)

print("\nStarting training (CPU-friendly)...")
trainer.train()
print("\nTraining completed.")


# ----------------------
# 8. Inference Demo
# ----------------------
classifier = pipeline(
    "sentiment-analysis",
    model="./sentiment_model/checkpoint-50",
    tokenizer=tokenizer,
    device=-1  # CPU
)

sentence = input("Please enter a sentence: ")
result = classifier(sentence)[0]
print(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})")
