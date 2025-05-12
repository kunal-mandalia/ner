from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Step 1: Load pre-trained model
model_name = "ProsusAI/finbert"  # or your fine-tuned NER model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Step 2: NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Step 3: Run on input text
text = "The company reported a net income of $2.5 million, an 8% increase from last year."
entities = ner_pipeline(text)

# Step 4: Post-process to match metrics to values
for ent in entities:
    print(ent)

# Optional: fine-tune a span relation classifier for metric ↔ value linking
