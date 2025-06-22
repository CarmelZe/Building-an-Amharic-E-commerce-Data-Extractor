from telethon.sync import TelegramClient
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv

load_dotenv()

client = TelegramClient('session_name', os.getenv('API_ID'), os.getenv('API_HASH')).start(os.getenv('PHONE'))

def extract_text_from_image(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang='amh')
    except:
        return ""

# Example: Download and OCR images from a channel
async def process_media(channel_name):
    os.makedirs(f'../generatedData/media/{channel_name}', exist_ok=True)
    async for message in client.iter_messages(channel_name, limit=100):
        if message.photo:
            file_path = f'../generatedData/media/{channel_name}/{message.id}.jpg'
            await client.download_media(message, file=file_path)
            text = extract_text_from_image(file_path)
            print(f"Extracted text: {text[:50]}...")  # Preview first 50 chars

# Run for a channel
with client:
    client.loop.run_until_complete(process_media('@ZemenExpress'))