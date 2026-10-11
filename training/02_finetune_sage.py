
from pathlib import Path
import shutil

import torch
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

#project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_sentiment_conversations.jsonl"
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loading dataset from: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {DATA_PATH}"
    )

#GPU availability
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. In Google Colab, go to "
        "Runtime > Change runtime type > T4 GPU, then rerun."
    )

print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")


#load and split dataset
dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train",
)

print(f"\nLoaded {len(dataset):,} examples")

if len(dataset) < 2:
    raise ValueError("Not enough examples to split the dataset.")

split_dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

print(f"Training examples: {len(train_dataset):,}")
print(f"Validation examples: {len(validation_dataset):,}")

assert "messages" in train_dataset.column_names
assert "messages" in validation_dataset.column_names

for example in [train_dataset[0], validation_dataset[0]]:
    messages = example["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] in {"BULLISH", "BEARISH"}

print("\nDataset format verification passed!")
print("\nExample:")
print(train_dataset[0]["messages"])


#load base model and tokenizer
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"\nBase model: {MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.chat_template is None:
    raise ValueError(
        "The selected model does not provide a chat template."
    )

print("Qwen chat template is available.")

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


#configure Lora
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

#configure training
FINAL_MODEL_DIR = (
    PROJECT_ROOT / "sentiment-classifier" / "final-adapter"
)

training_args = SFTConfig(
    output_dir=str(PROJECT_ROOT / "sentiment-classifier"),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    bf16=False,
)

print("\nSFT training configuration created!")

#Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
)

print("\nSFT trainer created successfully!")

# train 
print("\nStarting training...")
trainer.train()

print("\nTraining completed!")

#save
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

print(f"\nFinal Sage adapter saved to: {FINAL_MODEL_DIR}")

if not (FINAL_MODEL_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(
        "Adapter configuration was not saved successfully."
    )

if not (FINAL_MODEL_DIR / "adapter_model.safetensors").exists():
    raise FileNotFoundError(
        "Adapter weights were not saved successfully."
    )

print("Adapter files verified successfully!")



#zip backup in colab
BACKUP_BASE = "/content/sage-final-adapter"
BACKUP_PATH = f"{BACKUP_BASE}.zip"

# Remove a previous ZIP so the backup contains the latest adapter.
Path(BACKUP_PATH).unlink(missing_ok=True)

shutil.make_archive(
    BACKUP_BASE,
    "zip",
    root_dir=str(FINAL_MODEL_DIR),
)

if not Path(BACKUP_PATH).exists():
    raise FileNotFoundError("The adapter ZIP was not created.")

print(f"\nBackup ZIP created: {BACKUP_PATH}")
print(f"ZIP size: {Path(BACKUP_PATH).stat().st_size / (1024 * 1024):.2f} MB")

#backup
from google.colab import files

print("\nStarting adapter ZIP download...")
files.download(BACKUP_PATH)

print("\nSage training and backup process completed!")
