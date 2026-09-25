from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset from huggings pace
dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment",
    split="train"
)

print(f"Total headlines: {len(dataset):,}")

assert {"text", "label"}.issubset(dataset.column_names)

print("Dataset columns:", dataset.column_names)

print("\nFirst example:")
print(dataset[0])

# Separate the two classes
bearish = dataset.filter(lambda x: x["label"] == 0)
bullish = dataset.filter(lambda x: x["label"] == 1)

print(f"\nBearish headlines: {len(bearish):,}")
print(f"Bullish headlines: {len(bullish):,}")

n = min(500, len(bearish), len(bullish))

bearish_sample = bearish.shuffle(seed=42).select(range(n))
bullish_sample = bullish.shuffle(seed=42).select(range(n))

balanced = concatenate_datasets(
    [bearish_sample, bullish_sample]
).shuffle(seed=42)

print(f"Balanced dataset size: {len(balanced):,}")

def create_conversation(example):
    headline = (example.get("text") or "").strip()

    if not headline:
        raise ValueError("Empty headline encountered")

    user_msg = (
        "Classify the market sentiment of this financial news headline:\n"
        f"{headline}"
    )

    asst_msg = "BULLISH" if example["label"] == 1 else "BEARISH"

    return {
        "messages": [
            {
                "role": "user",
                "content": user_msg
            },
            {
                "role": "assistant",
                "content": asst_msg
            }
        ]
    }

train_dataset = balanced.map(create_conversation)

print(f"\nCreated {len(train_dataset):,} training conversations")

print("\nExample conversation:")
print(train_dataset[0]["messages"])

assert len(train_dataset) == 2 * n

messages = train_dataset[0]["messages"]

assert len(messages) == 2
assert messages[0]["role"] == "user"
assert messages[1]["role"] == "assistant"
assert messages[1]["content"] in {"BULLISH", "BEARISH"}

print("\nVerification passed!")

balanced_path = DATA_DIR / "market_sentiment_balanced.jsonl"
conversations_path = DATA_DIR / "market_sentiment_conversations.jsonl"


# Save balanced dataset
with open(balanced_path, "w", encoding="utf-8") as f:
    for example in balanced:
        f.write(json.dumps(example) + "\n")


# Save conversational training dataset
with open(conversations_path, "w", encoding="utf-8") as f:
    for example in train_dataset:
        f.write(json.dumps(example) + "\n")


print("\nDatasets saved successfully!")
print(f"Balanced dataset: {balanced_path}")
print(f"Conversation dataset: {conversations_path}")