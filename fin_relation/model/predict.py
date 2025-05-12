import os
from pathlib import Path
from typing import List, Dict, Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class FinancialRelationPredictor:
    def __init__(self, model_path: Union[str, Path] = "./fin_relation_model"):
        self.model_name = "ProsusAI/finbert"
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
    
    def predict(self, text: str) -> Dict:
        """Make a prediction for a single text."""
        result = self.classifier(text)
        return {
            "text": text,
            "is_related": result[0]["label"] == "LABEL_1",
            "confidence": result[0]["score"]
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Make predictions for multiple texts."""
        return [self.predict(text) for text in texts]

def main():
    # Example usage
    predictor = FinancialRelationPredictor()
    
    test_sentences = [
        "The company reported a <metric>net income</metric> of <value>$2.5 million</value>.",
        "Operating margin remained steady while <value>$2.5 million</value> was allocated to R&D.",
        "Revenue increased by <value>15%</value> due to strong <metric>sales growth</metric>.",
        "The <metric>profit margin</metric> was <value>25%</value> last quarter."
    ]
    
    results = predictor.predict_batch(test_sentences)
    
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Prediction: {'Related' if result['is_related'] else 'Not Related'} "
              f"(confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    main() 