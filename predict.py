from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the tokenizer from the original model and the fine-tuned model from the output directory
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./fin_relation_model")

# Create a classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test sentences
test_sentences = [
    "The company reported a <metric>net income</metric> of <value>$2.5 million</value>.",
    "Operating margin remained steady while <value>$2.5 million</value> was allocated to R&D.",
    "Revenue increased by <value>15%</value> due to strong <metric>sales growth</metric>.",
    "The <metric>profit margin</metric> was <value>25%</value> last quarter."
]

# Make predictions
for sentence in test_sentences:
    result = classifier(sentence)
    print(f"\nText: {sentence}")
    print(f"Prediction: {'Related' if result[0]['label'] == 'LABEL_1' else 'Not Related'} (confidence: {result[0]['score']:.2f})") 