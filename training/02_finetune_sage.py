from pathlib import Path

from datasets import load_dataset

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig

from trl import SFTConfig, SFTTrainer


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to our processed training data
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_sentiment_conversations.jsonl"
)


print(f"Loading dataset from: {DATA_PATH}")

train_dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train",
)

print(f"Loaded {len(train_dataset):,} training examples")

print("\nFirst example:")
print(train_dataset[0])


split_dataset = train_dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

print(f"\nTraining examples: {len(train_dataset):,}")
print(f"Validation examples: {len(validation_dataset):,}")


print("\nTraining example:")
print(train_dataset[0]["messages"])

print("\nValidation example:")
print(validation_dataset[0]["messages"])

assert "messages" in train_dataset.column_names
assert "messages" in validation_dataset.column_names

assert len(train_dataset[0]["messages"]) == 2
assert len(validation_dataset[0]["messages"]) == 2

assert train_dataset[0]["messages"][0]["role"] == "user"
assert train_dataset[0]["messages"][1]["role"] == "assistant"

assert validation_dataset[0]["messages"][0]["role"] == "user"
assert validation_dataset[0]["messages"][1]["role"] == "assistant"

assert train_dataset[0]["messages"][1]["content"] in {
    "BULLISH",
    "BEARISH",
}

assert validation_dataset[0]["messages"][1]["content"] in {
    "BULLISH",
    "BEARISH",
}

print("\nDataset format verification passed!")


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"\nBase model: {MODEL_ID}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)



tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Verify that the tokenizer has a chat template
if tokenizer.chat_template is None:
    raise ValueError(
        "The selected model does not provide a chat template."
    )

print("\nQwen chat template is available.")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

print("\nModel and tokenizer loaded successfully!")



formatted_prompt = tokenizer.apply_chat_template(
    train_dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print("\nFormatted example:")
print(formatted_prompt)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("\nLoRA configuration created!")


training_args = SFTConfig(
    output_dir="./sentiment-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

print("\nSFT training configuration created!")


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
)

print("\nSFT trainer created successfully!")

print("\nStarting training...")
trainer.train()