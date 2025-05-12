from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Example dataset
examples = [
    {
        "text": "The company reported a <metric>net income</metric> of <value>$2.5 million</value>.",
        "label": 1,
    },
    {
        "text": "Operating margin remained steady while <value>$2.5 million</value> was allocated to R&D.",
        "label": 0,
    },
]

# Load tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# Tokenize
def preprocess(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = example["label"]
    return encoding

dataset = Dataset.from_list(examples).map(preprocess)

# Training setup
training_args = TrainingArguments(
    output_dir="./fin_relation_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save the model and tokenizer in Hugging Face format
model.save_pretrained("./fin_relation_model")
tokenizer.save_pretrained("./fin_relation_model")
