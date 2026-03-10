import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Legacy Euri AI Configuration 
EURI_API_KEY = os.getenv("EURI_API_KEY")

# Model Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
    'smtp_port': int(os.getenv("EMAIL_SMTP_PORT", "587")),
    'sender_email': os.getenv("EMAIL_SENDER", "your_email@gmail.com"),
    'sender_password': os.getenv("EMAIL_PASSWORD", "your_app_password"),
    'receiver_email': os.getenv("EMAIL_RECEIVER", "sudhanshu@euron.on")
}