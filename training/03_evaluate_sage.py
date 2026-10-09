
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_sentiment_conversations.jsonl"
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# The final checkpoint created after 3 training epochs.
ADAPTER_PATH = PROJECT_ROOT / "sentiment-classifier" / "checkpoint-339"

print("Sage evaluation script initialized.")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset path: {DATA_PATH}")
print(f"Base model: {MODEL_ID}")
print(f"Adapter path: {ADAPTER_PATH}")


##load dataset
dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train",
)

split_dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

validation_dataset = split_dataset["test"]

print(f"\nValidation examples: {len(validation_dataset)}")

# Check that the dataset has the expected format.
assert len(validation_dataset) == 100
assert "messages" in validation_dataset.column_names

print("Validation dataset loaded successfully!")


#load base model 
from transformers import BitsAndBytesConfig

# Evaluation must run on the Colab GPU.
assert torch.cuda.is_available(), "Please use a GPU runtime in Google Colab."

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
)

model.eval()

print("\nSage's trained adapter loaded successfully!")


#testing
example = validation_dataset[0]

user_message = example["messages"][0]

inputs = tokenizer.apply_chat_template(
    [user_message],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only the newly generated answer.
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

prediction = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
).strip().upper()

expected = example["messages"][1]["content"].strip().upper()

print("\nSAGE TEST")
print("Headline:", user_message["content"])
print("Expected sentiment:", expected)
print("Sage's prediction:", prediction)


#evaluation
correct = 0
incorrect = 0
invalid = 0

confusion = {
    "BEARISH": {"BEARISH": 0, "BULLISH": 0},
    "BULLISH": {"BEARISH": 0, "BULLISH": 0},
}

for index, example in enumerate(validation_dataset):
    user_message = example["messages"][0]
    expected = example["messages"][1]["content"].strip().upper()

    inputs = tokenizer.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip().upper()

    if "BULLISH" in answer and "BEARISH" not in answer:
        predicted = "BULLISH"
    elif "BEARISH" in answer and "BULLISH" not in answer:
        predicted = "BEARISH"
    else:
        predicted = "INVALID"

    if predicted == expected:
        correct += 1
    elif predicted == "INVALID":
        invalid += 1
    else:
        incorrect += 1

    if predicted != "INVALID":
        confusion[expected][predicted] += 1

    print(
        f"{index + 1:03d}/100 | "
        f"Expected: {expected:7s} | "
        f"Predicted: {predicted}"
    )

total = len(validation_dataset)
accuracy = correct / total * 100

print("\nFINAL EVALUATION")
print(f"Total examples: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Invalid predictions: {invalid}")
print(f"Accuracy: {accuracy:.2f}%")
print("\nConfusion matrix:")
print("                 Predicted BEARISH | Predicted BULLISH")
print(
    f"Actual BEARISH:       "
    f"{confusion['BEARISH']['BEARISH']:3d}       | "
    f"{confusion['BEARISH']['BULLISH']:3d}"
)
print(
    f"Actual BULLISH:       "
    f"{confusion['BULLISH']['BEARISH']:3d}       | "
    f"{confusion['BULLISH']['BULLISH']:3d}"
)
