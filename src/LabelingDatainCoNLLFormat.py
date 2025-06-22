import pandas as pd
from pathlib import Path

def load_and_prepare_data(input_csv: str, num_messages: int = 50) -> list:
    """
    Load preprocessed data and extract text messages.
    Args:
        input_csv: Path to the CSV file from Task 1
        num_messages: Number of messages to label
    Returns:
        List of text messages
    """
    try:
        df = pd.read_csv(input_csv)
        
        # Try common column names that might contain text
        text_col = None
        for col in ['amharic_text', 'text', 'cleaned_text', 'message']:
            if col in df.columns:
                text_col = col
                break
                
        if not text_col:
            raise ValueError("No suitable text column found. Expected one of: 'amharic_text', 'text', 'cleaned_text'")
            
        messages = df[text_col].dropna().tolist()[:num_messages]
        print(f"Loaded {len(messages)} messages from column '{text_col}'")
        return messages
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Available columns: {list(df.columns) if 'df' in locals() else 'File not loaded'}")
        return []

def label_message_interactive(tokens: list) -> list:
    """
    Helper function to manually label tokens with entity tags.
    Args:
        tokens: List of words from a message
    Returns:
        List of labels corresponding to each token
    """
    print("\nMessage to label:")
    print(" ".join(tokens))
    
    labels = []
    tag_options = {
        '1': 'O',
        '2': 'B-PRODUCT',
        '3': 'I-PRODUCT',
        '4': 'B-PRICE',
        '5': 'I-PRICE',
        '6': 'B-LOC',
        '7': 'I-LOC'
    }
    
    print("\nLabeling Guide:")
    print("1: O (Other)\n2: B-PRODUCT\n3: I-PRODUCT\n4: B-PRICE\n5: I-PRICE\n6: B-LOC\n7: I-LOC")
    
    for i, token in enumerate(tokens):
        while True:
            choice = input(f"Token '{token}': Enter label (1-7): ")
            if choice in tag_options:
                labels.append(tag_options[choice])
                break
            print("Invalid choice. Try 1-7.")
    return labels

def generate_connl_file(messages: list, output_file: str = 'amharic_ner.conll'):
    """
    Generate CoNLL format file from labeled messages.
    Args:
        messages: List of raw message texts
        output_file: Path to save the CoNLL file
    """
    labeled_data = []
    
    for msg in messages:
        if not isinstance(msg, str):
            continue
            
        tokens = msg.split()
        if not tokens:
            continue
            
        print(f"\n{'='*40}\nLabeling new message ({len(tokens)} tokens)")
        labels = label_message_interactive(tokens)
        
        # Add to CoNLL data
        for token, label in zip(tokens, labels):
            labeled_data.append(f"{token}\t{label}")
        labeled_data.append("")  # Blank line between messages
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(labeled_data))
    print(f"\nSuccess! Saved labeled data to {output_file}")

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = '../generatedData/preprocessed_data.csv'  # Update this path
    OUTPUT_FILE = 'amharic_ner.conll'
    NUM_MESSAGES = 5  # Adjust as needed
    
    # Step 1: Load data
    print(f"Looking for data in: {Path(INPUT_CSV).absolute()}")
    messages = load_and_prepare_data(INPUT_CSV, NUM_MESSAGES)
    
    if messages:
        # Step 2: Interactive labeling
        generate_connl_file(messages, OUTPUT_FILE)
    else:
        print("No messages loaded. Please check:")
        print(f"1. File exists at: {INPUT_CSV}")
        print("2. CSV contains one of these columns: 'amharic_text', 'text', 'cleaned_text'")
        print("3. File has valid Amharic text data")