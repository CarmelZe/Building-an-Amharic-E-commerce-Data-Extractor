import re
import pandas as pd
from tqdm import tqdm

# Load raw data
df = pd.read_csv('../generatedData/raw_telegram_data.csv')

# Text cleaning function for Amharic
def clean_amharic_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs, mentions, and special chars (keep Amharic Unicode range)
    text = re.sub(r'http\S+|@\w+|[^\w\s\u1200-\u137F]', '', text)
    
    # Normalize currency (e.g., "ብር" → "ETB")
    text = re.sub(r'ብር|birr', 'ETB', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

# Apply cleaning
tqdm.pandas()
df['cleaned_text'] = df['text'].progress_apply(clean_amharic_text)

# Tokenization (simple whitespace-based)
df['tokens'] = df['cleaned_text'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Save preprocessed data
df.to_csv('../generatedData/preprocessed_data.csv', index=False)
print("Preprocessed data saved to preprocessed_data.csv")