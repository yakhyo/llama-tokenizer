import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from typing import List
# Check if a GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------

# Load the LLaMA 3.1 tokenizer and model
tokenizer_name = "new-llama-tokenizer"
model_name = "meta-llama/Llama-3.2-1b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#-------------------------------------------------------------------------------

def prepare_datasets(datasets_list: List[str]):
    all_data = []
    for dataset_name in datasets_list:
        try:
            data = load_dataset(dataset_name)
            for split in ["train", "test", "validation"]:
                try:
                    all_data.append(data[split])
                except KeyError:
                    pass
        except:
            print(f"dataset: `{dataset_name}` not found, skipping...")

    concat_data = []
    for data in all_data:
        data = data.remove_columns([col for col in data.column_names if col != "text"])
        concat_data.append(data)

    return concatenate_datasets(concat_data)

#-------------------------------------------------------------------------------

# Load your dataset (replace with your dataset path)
hf_datasets = ["yakhyo/uz-wiki", "yakhyo/uz-news"]

dataset = prepare_datasets(hf_datasets)
split_dataset = dataset.train_test_split(test_size=0.1)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

#-------------------------------------------------------------------------------

# Data collator for batching the data
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments - adjusted for single-device setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,  # Adjusted for smaller memory
    per_device_eval_batch_size=2,   # Adjusted for smaller memory
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    push_to_hub=False,  # Disable this if not using Hugging Face Hub
    fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
    gradient_accumulation_steps=8,  # Use gradient accumulation for smaller batches
)


#-------------------------------------------------------------------------------


# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./llama-3.2-1b")

