#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amharic NER Model Interpretability with SHAP and LIME
Task 5 for EthioMart E-commerce Data Extractor Project
"""

import numpy as np
import shap
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

class AmharicNERInterpretability:
    def __init__(self, model_path):
        """Initialize the interpretability analyzer with a fine-tuned NER model"""
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, 
                                    aggregation_strategy="simple")
        self.setup_shap_explainer()
        self.setup_lime_explainer()
        
    def setup_shap_explainer(self):
        """Configure the SHAP explainer for the NER model"""
        def f(x):
            outputs = []
            for text in x:
                tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                predictions = self.model(**tokens).logits.argmax(-1).squeeze().tolist()
                words = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
                outputs.append([(word, pred) for word, pred in zip(words, predictions)])
            return np.array(outputs)
        
        self.shap_explainer = shap.Explainer(f, self.tokenizer)
    
    def setup_lime_explainer(self):
        """Configure the LIME explainer for the NER model"""
        def predict_proba(texts):
            results = []
            for text in texts:
                ner_results = self.ner_pipeline(text)
                dummy_probs = np.zeros((1, len(self.model.config.id2label)))
                for entity in ner_results:
                    label_id = self.model.config.label2id[entity['entity_group']]
                    dummy_probs[0, label_id] = 1
                results.append(dummy_probs[0])
            return np.array(results)
        
        self.lime_explainer = LimeTextExplainer(
            class_names=list(self.model.config.id2label.values()))
        self.lime_predict_proba = predict_proba
    
    def analyze_with_shap(self, text):
        """Analyze text with SHAP and return visualization"""
        shap_values = self.shap_explainer([text])
        return shap.plots.text(shap_values, display=False)
    
    def analyze_with_lime(self, text):
        """Analyze text with LIME and return explanation"""
        exp = self.lime_explainer.explain_instance(
            text, self.lime_predict_proba, num_features=10, top_labels=3)
        return exp.as_html()
    
    def generate_interpretability_report(self, text_samples, output_file="ner_interpretability_report.html"):
        """Generate a comprehensive HTML report with SHAP and LIME analyses"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NER Model Interpretability Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        h3 { color: #16a085; }
        .sample { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .predictions { background: #e8f4f8; padding: 10px; border-radius: 5px; }
        .entity { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Amharic NER Model Interpretability Report</h1>
    <p>Generated for EthioMart E-commerce Data Extractor Project</p>
""")
            
            for i, text in enumerate(text_samples):
                f.write(f"""
    <div class="sample">
        <h2>Sample {i+1}</h2>
        <p><strong>Original Text:</strong> {text}</p>
        
        <div class="predictions">
            <h3>Model Predictions</h3>
            <ul>
""")
                
                # Get model predictions
                predictions = self.ner_pipeline(text)
                for pred in predictions:
                    f.write(f'<li><span class="entity">{pred["word"]}</span> → {pred["entity_group"]} '
                            f'(confidence: {pred["score"]:.2f})</li>\n')
                
                f.write("""
            </ul>
        </div>
        
        <div class="shap-analysis">
            <h3>SHAP Analysis</h3>
""")
                
                # Add SHAP visualization
                shap_html = self.analyze_with_shap(text)
                f.write(shap_html)
                
                f.write("""
        </div>
        
        <div class="lime-analysis">
            <h3>LIME Analysis</h3>
""")
                
                # Add LIME visualization
                lime_html = self.analyze_with_lime(text)
                f.write(lime_html)
                
                f.write("""
        </div>
    </div>
""")
            
            f.write("""
</body>
</html>
""")
        
        print(f"Successfully generated interpretability report: {output_file}")

def main():
    # Example usage
    MODEL_PATH = "../model/amharic_ner_task3.ipynb"  # Replace with your actual model path
    
    # Sample Amharic texts for analysis
    sample_texts = [
        "ቤት አስፋልት በቦሌ በ250,000 ብር ይሸጣል",  # "House for sale in Bole for 250,000 birr"
        "አዲስ ስልክ ሳምሱንግ በ15,000 ብር አለኝ",  # "New Samsung phone available for 15,000 birr"
        "የቤት እቃዎች በሰሜን አዲስ አበባ ይገኛሉ",  # "Home appliances available in North Addis Ababa"
        "በ200 ብር አዲስ አበባ ውስጥ በ200 ብር ልብስ ይገኛል",  # Ambiguous case
        "ሳምሱንግ ጋላክሲ ኤስ20 በቦሌ ሚካኤል ሻንጣ"  # Complex entity case
    ]
    
    # Initialize interpretability analyzer
    print("Initializing Amharic NER Interpretability Analyzer...")
    analyzer = AmharicNERInterpretability(MODEL_PATH)
    
    # Generate comprehensive report
    print("Generating interpretability report...")
    analyzer.generate_interpretability_report(sample_texts)
    
    print("Interpretability analysis completed successfully!")

if __name__ == "__main__":
    main()