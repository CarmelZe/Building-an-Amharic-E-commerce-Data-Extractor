import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification,
    pipeline
)
from datasets import Dataset, DatasetDict
from huggingface_hub import notebook_login
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm.auto import tqdm

# ===== 1. ENVIRONMENT SETUP =====
def setup_environment():
    """Verify dependencies and hardware"""
    print("Setting up environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Using CPU - training will be slow")

# ===== 2. DATA LOADING & VALIDATION =====
class DataLoader:
    def __init__(self, file_path="amharic_ner.conll"):
        self.file_path = Path(file_path)
        self.validate_file()
        
    def validate_file(self):
        """Verify CoNLL file exists and is readable"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CoNLL file not found at: {self.file_path.absolute()}")
        print(f"Found CoNLL file at: {self.file_path.absolute()}")
        
        # Quick format check
        with open(self.file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(5) if f.readline().strip()]
        if not any(len(line.split()) > 1 for line in first_lines):
            print("WARNING: File may not be in proper CoNLL format (missing labels)")
    
    def load_data(self):
        """Load and validate CoNLL data"""
        sentences = []
        current_sentence = []
        label_stats = Counter()
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading CoNLL file"):
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    token = parts[0]
                    label = parts[-1] if len(parts) > 1 else "O"
                    current_sentence.append((token, label))
                    label_stats[label] += 1
        
        if current_sentence:
            sentences.append(current_sentence)
            
        print("\nLabel distribution:")
        for label, count in label_stats.most_common():
            print(f"{label}: {count} ({count/sum(label_stats.values()):.1%})")
        
        return sentences
    
    def create_datasets(self, test_size=0.2):
        """Create train/validation datasets"""
        sentences = self.load_data()
        data = []
        
        for sentence in sentences:
            tokens = [token for token, label in sentence]
            ner_tags = [label for token, label in sentence]
            data.append({'tokens': tokens, 'ner_tags': ner_tags})
            
        train_df, val_df = train_test_split(
            pd.DataFrame(data), 
            test_size=test_size, 
            random_state=42,
            stratify=[len(x) for x in data]  # Maintain similar length distribution
        )
        
        return DatasetDict({
            'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
            'validation': Dataset.from_pandas(val_df.reset_index(drop=True))
        })

# ===== 3. MODEL TRAINING =====
class NERTrainer:
    def __init__(self, dataset, label_list):
        self.dataset = dataset
        self.label_list = label_list
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.models = {
            "xlm-roberta": "xlm-roberta-base",
            "distilbert": "distilbert-base-multilingual-cased", 
            "mbert": "bert-base-multilingual-cased"
        }
        
    def tokenize_and_align_labels(self, examples, tokenizer):
        """Align labels with tokenized inputs"""
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            max_length=512,
            padding='max_length'
        )
        labels = []
        
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            # Ensure length matches max_length
            label_ids += [-100] * (512 - len(label_ids))
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, p):
        """Calculate precision, recall, and F1"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "classification_report": classification_report(
                true_labels, 
                true_predictions,
                output_dict=True
            )
        }
    
    def train_model(self, model_name, model_path):
        """Train and evaluate a single model"""
        print(f"\n=== Training {model_name} ===")
        
        try:
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # Tokenize datasets
            tokenized_datasets = self.dataset.map(
                lambda x: self.tokenize_and_align_labels(x, tokenizer),
                batched=True,
                remove_columns=['tokens', 'ner_tags']
            )
            
            data_collator = DataCollatorForTokenClassification(tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{model_name}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=3e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                logging_dir=f'./logs/{model_name}',
                logging_steps=50,
                report_to="none",
                save_total_limit=2,
                fp16=torch.cuda.is_available()
            )
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            # Train and evaluate
            trainer.train()
            eval_results = trainer.evaluate()
            
            # Measure inference speed
            test_text = " ".join(self.dataset['validation'][0]['tokens'][:10])
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            pipe = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            
            # Warmup
            for _ in range(3):
                pipe(test_text)
            
            # Timed inference
            start.record()
            results = pipe(test_text)
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end) / 1000
            
            return {
                "model": model_name,
                "f1": eval_results["eval_f1"],
                "precision": eval_results["eval_precision"],
                "recall": eval_results["eval_recall"],
                "inference_time": inference_time,
                "classification_report": eval_results["eval_classification_report"],
                "trainer": trainer,
                "tokenizer": tokenizer
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return None
    
    def compare_models(self):
        """Train and compare all models"""
        results = []
        
        for name, path in tqdm(self.models.items(), desc="Comparing models"):
            result = self.train_model(name, path)
            if result:
                results.append(result)
                
        return results

# ===== 4. MAIN EXECUTION =====
def main():
    setup_environment()
    
    # Initialize data loader
    loader = DataLoader()
    dataset = loader.create_datasets()
    
    # Define labels (update with your actual labels)
    label_list = ["O", "B-PRODUCT", "I-PRODUCT", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
    
    # Train and compare models
    trainer = NERTrainer(dataset, label_list)
    results = trainer.compare_models()
    
    if results:
        # Save comparison results
        results_df = pd.DataFrame([{
            'model': r['model'],
            'f1': r['f1'],
            'precision': r['precision'],
            'recall': r['recall'],
            'inference_time': r['inference_time']
        } for r in results])
        
        print("\n=== Model Comparison Results ===")
        print(results_df.sort_values('f1', ascending=False))
        
        # Save best model
        best_result = max(results, key=lambda x: x['f1'])
        save_path = f"./best_model_{best_result['model']}"
        
        best_result['trainer'].save_model(save_path)
        best_result['tokenizer'].save_pretrained(save_path)
        
        print(f"\nSaved best model ({best_result['model']}) to {save_path}")
        print("Classification Report:")
        print(pd.DataFrame(best_result['classification_report']).transpose())
        
        # Optional: Push to Hub
        # notebook_login()
        # best_result['trainer'].push_to_hub("your-username/amharic-ner-best")
    else:
        print("No models were successfully trained")

if __name__ == "__main__":
    main()