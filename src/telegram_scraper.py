import os
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerChannel
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Telegram API credentials
api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE')

# Target channels (from your list)
channels = [
    '@ZemenExpress',
    '@nevacomputer',
    '@meneshayeofficial',
    '@ethio_brand_collection',
    '@sinayelj'
]

# Initialize client
client = TelegramClient('session_name', api_id, api_hash).start(phone)

# Scrape messages
messages = []
for channel in channels:
    try:
        print(f"Scraping {channel}...")
        for message in client.iter_messages(channel, limit=1000):  # Adjust limit as needed
            messages.append({
                'channel': channel,
                'sender_id': message.sender_id,
                'date': message.date,
                'text': message.text,
                'views': message.views if hasattr(message, 'views') else None,
                'media_type': 'photo' if message.photo else 'document' if message.document else 'text'
            })
        time.sleep(2)  # Avoid rate limiting
    except Exception as e:
        print(f"Error scraping {channel}: {str(e)}")

# Save to CSV
df = pd.DataFrame(messages)
df.to_csv('../generatedData/raw_telegram_data.csv', index=False)
print("Raw data saved to raw_telegram_data.csv")