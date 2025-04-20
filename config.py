# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys & Supabase Storage ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Service Role Key for Storage

# --- Supabase Database Credentials (for psycopg2) ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# --- Model Names ---
GROQ_LLM_MODEL = "llama3-70b-8192"
GEMINI_VISION_MODEL = "gemini-1.5-flash-latest"
GROQ_WHISPER_MODEL = "whisper-large-v3"

# --- Supabase Config ---
SUPABASE_BUCKET_NAME = "receipts"

# --- Agent Config ---
MAX_SQL_RETRIES = 3

# --- Validation ---
# Updated validation section
required_vars = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
}
missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    raise ValueError(f"Missing essential environment variables: {', '.join(missing_vars)}. Please check your .env file.")


# --- Database Schema Info (Remains the same conceptually) ---
DB_SCHEMA_INFO = """
Supabase Public Schema (PostgreSQL):

Tables:
1. employees
   Columns:
     - id (integer, auto-generated, primary key)
     - created_at (timestamp with time zone, auto-generated)
     - name (text)
     - age (integer)
     - salary (integer)

2. refund_requests
   Columns:
     - id (integer, auto-generated, primary key)
     - created_at (timestamp with time zone, auto-generated)
     - name (text)
     - amount (numeric/float)
     - image_url (text, public URL to Supabase Storage)
     - audio_url (text, public URL to Supabase Storage)

Storage Bucket:
- receipts (public read access required for image/audio tasks)
  Example URL format: {SUPABASE_URL}/storage/v1/object/public/receipts/YOUR_FILE_NAME.ext
"""

# Construct connection string (optional, but can be useful)
DATABASE_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"