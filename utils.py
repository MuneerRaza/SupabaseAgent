# utils.py
import config
import datetime
import requests
import logging
import psycopg2
from psycopg2.extras import RealDictCursor # To get results as dictionaries
from supabase import create_client, Client # Keep for Storage
from groq import Groq
from openai import OpenAI as OpenAI_WhisperClient
import io
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any, Optional

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Clients ---

# Supabase Client (ONLY for Storage interactions now)
try:
    supabase_storage_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    logging.info("Supabase client initialized successfully (for Storage).")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client (for Storage): {e}")
    supabase_storage_client = None # Storage functions will fail if this is None

# Groq Client
try:
    groq_client = Groq(api_key=config.GROQ_API_KEY)
    logging.info("Groq client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

# Groq Whisper Client
try:
    whisper_client = OpenAI_WhisperClient(
        api_key=config.GROQ_API_KEY,
        base_url='https://api.groq.com/openai/v1'
    )
    logging.info("Groq Whisper client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Groq Whisper client: {e}")
    whisper_client = None

# ---- Configure Gemini ---
try:
    if config.GOOGLE_API_KEY:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        logging.info("Google Generative AI SDK configured successfully.")
        # Optional: Test configuration by listing models
        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)
    else:
        logging.warning("GOOGLE_API_KEY not found. Gemini Vision functionality will be unavailable.")
except Exception as e:
    logging.error(f"Failed to configure Google Generative AI SDK: {e}", exc_info=True)
#

# --- Database Functions (using psycopg2) ---

def execute_sql(query: str) -> Dict[str, Any]:
    """Executes a SQL query against the Supabase database using psycopg2."""
    conn = None
    try:
        logging.info(f"Connecting to database: {config.DB_HOST}...")
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        conn.autocommit = False # Use transactions
        logging.info("Database connection successful.")
        # Use RealDictCursor to get results as dictionaries
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        logging.info(f"Executing SQL: {query}")
        cursor.execute(query)

        # Check if the query was a SELECT statement to fetch results
        # A more robust way might involve parsing the query type, but this is simpler
        data = None
        if cursor.description: # SELECT statements have descriptions
            data = cursor.fetchall()
            logging.info(f"SQL Result Data Count: {len(data)}")
            # Convert RealDictRow objects to plain dicts for broader compatibility
            data = [dict(row) for row in data]
            # Convert datetime objects to ISO format strings for JSON serialization
            for row in data:
                for key, value in row.items():
                    if isinstance(value, datetime.datetime):
                        row[key] = value.isoformat()
        else:
            # For INSERT, UPDATE, DELETE, etc.
            rowcount = cursor.rowcount
            logging.info(f"SQL executed successfully. Rows affected: {rowcount}")
            data = f"Command executed successfully. {rowcount} rows affected." # Or simply return rowcount

        conn.commit() # Commit transaction
        cursor.close()
        logging.info("SQL execution successful and transaction committed.")
        return {"data": data, "error": None}

    except (Exception, psycopg2.Error) as error:
        logging.error(f"Error during SQL execution: {error}", exc_info=True)
        if conn:
            try:
                conn.rollback() # Rollback transaction on error
                logging.info("Transaction rolled back.")
            except Exception as rb_error:
                logging.error(f"Error during rollback: {rb_error}", exc_info=True)
        return {"error": f"Database Error: {str(error)}"}

    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


# --- Supabase Storage Functions (using supabase-py client) ---

def get_public_url(file_name: str) -> str:
    """Constructs the public URL for a file in Supabase Storage."""
    if not supabase_storage_client: # Use the dedicated storage client
        logging.error("Cannot get public URL: Supabase storage client not initialized.")
        return ""
    try:
        url = supabase_storage_client.storage.from_(config.SUPABASE_BUCKET_NAME).get_public_url(file_name)
        logging.info(f"Generated public URL for {file_name}: {url}")
        return url
    except Exception as e:
        logging.error(f"Error getting public URL for {file_name}: {e}")
        return ""

def list_storage_files(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists files in the Supabase storage bucket."""
    if not supabase_storage_client: # Use the dedicated storage client
        logging.error("Cannot list storage files: Supabase storage client not initialized.")
        return []
    try:
        logging.info(f"Listing files in bucket '{config.SUPABASE_BUCKET_NAME}' at path: '{path or '/'}'")
        files = supabase_storage_client.storage.from_(config.SUPABASE_BUCKET_NAME).list(path=path)
        logging.info(f"Found {len(files)} files.")
        # Convert StorageFile objects to dicts if necessary for consistent state handling
        files = [file.__dict__ for file in files] if files and hasattr(files[0], '__dict__') else files
        return files
    except Exception as e:
        logging.error(f"Error listing storage files: {e}")
        return []

# --- LLM, VLM, Audio Functions (Remain unchanged) ---
# (Keep the existing call_groq_llama, call_groq_whisper, translate_text functions)

def call_groq_llama(prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> Optional[str]:
    """Calls the Groq LLaMA model."""
    if not groq_client:
        logging.error("Cannot call Groq LLaMA: Groq client not initialized.")
        return None
    try:
        logging.info(f"Calling Groq LLaMA with model: {config.GROQ_LLM_MODEL}")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=config.GROQ_LLM_MODEL,
            temperature=0.7,
            max_tokens=2048,
        )
        response_content = chat_completion.choices[0].message.content
        logging.info("Groq LLaMA call successful.")
        return response_content
    except Exception as e:
        logging.error(f"Error calling Groq LLaMA: {e}", exc_info=True)
        return 
    
def call_gemini_vision(image_url: str, prompt: str) -> Optional[str]:
    """Downloads an image and uses Gemini Vision to process it based on the prompt."""
    if not config.GOOGLE_API_KEY:
        logging.error("Cannot call Gemini Vision: GOOGLE_API_KEY not configured.")
        return None
    try:
        logging.info(f"Processing image with Gemini Vision: {image_url}")
        # Download image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_bytes = response.content

        # Prepare image for Gemini
        img = Image.open(io.BytesIO(image_bytes))

        # Initialize the Gemini model
        # Consider adding safety_settings if needed: safety_settings=[...]
        model = genai.GenerativeModel(config.GEMINI_VISION_MODEL)

        # Send prompt and image to the model
        # The SDK expects a list containing text and image parts
        logging.info(f"Calling Gemini Vision model ({config.GEMINI_VISION_MODEL}) with prompt: '{prompt}'")
        # Note: Explicitly setting stream=False for synchronous behavior
        response = model.generate_content([prompt, img], stream=False)

        # Check for safety blocks before accessing text
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logging.error(f"Gemini Vision call blocked. Reason: {block_reason}")
            # You might want to check safety_ratings too for finer detail
            # ratings = response.prompt_feedback.safety_ratings
            return f"Error: Content blocked by safety filter - {block_reason}"

        # Check if response has parts and text (structure might vary slightly)
        if hasattr(response, 'text'):
            result_text = response.text
            logging.info(f"Gemini Vision call successful. Result length: {len(result_text)}")
            # Clean potential markdown backticks if the model adds them despite the prompt
            if result_text.startswith("```json"):
                result_text = result_text.strip()[7:-3].strip() # Remove ```json ... ```
            elif result_text.startswith("```"):
                 result_text = result_text.strip()[3:-3].strip() # Remove ``` ... ```
            return result_text
        else:
            # Log the response structure if text isn't directly available
            logging.warning(f"Gemini Vision response did not contain text directly. Response: {response}")
            # Try accessing parts if available
            if response.parts:
                 result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                 if result_text:
                      logging.info("Extracted text from response parts.")
                      # Clean potential markdown here too
                      if result_text.startswith("```json"):
                         result_text = result_text.strip()[7:-3].strip()
                      elif result_text.startswith("```"):
                         result_text = result_text.strip()[3:-3].strip()
                      return result_text
            logging.error("Could not extract text from Gemini Vision response.")
            return None


    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download image for Gemini Vision from {image_url}: {e}")
        return None
    except FileNotFoundError: # If PIL tries to access a non-existent temp file (less likely with BytesIO)
        logging.error(f"Error opening image file (FileNotFound) for {image_url}", exc_info=True)
        return None
    except Exception as e:
        # Catch exceptions from the genai call specifically if possible
        logging.error(f"Error during Gemini Vision processing for {image_url}: {e}", exc_info=True)
        # Check if it's an API key error from genai (structure might vary)
        # if "API key not valid" in str(e):
        #     return "Error: Invalid Google API Key."
        return f"Error: An exception occurred during Gemini processing - {str(e)}"


def call_groq_whisper(audio_url: str) -> Optional[str]:
    """Calls Groq's Whisper API for audio transcription."""
    if not whisper_client:
        logging.error("Cannot call Groq Whisper: Whisper client not initialized.")
        return None
    try:
        logging.info(f"Calling Groq Whisper for URL: {audio_url}")
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        audio_content = response.content
        file_tuple = ("audio_from_url.mp3", audio_content) # Assume mp3
        transcription = whisper_client.audio.transcriptions.create(
            model=config.GROQ_WHISPER_MODEL,
            file=file_tuple,
        )
        logging.info("Groq Whisper call successful.")
        return transcription.text if hasattr(transcription, 'text') else str(transcription)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading audio file {audio_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling Groq Whisper for {audio_url}: {e}", exc_info=True)
        return None

def translate_text(text: str, target_language: str = "English") -> Optional[str]:
    """Translates text using Groq LLaMA."""
    logging.info(f"Translating text to {target_language}.")
    system_prompt = f"Translate the following text to {target_language}. Output only the translated text."
    return call_groq_llama(text, system_prompt=system_prompt)


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing utils functions...")
    # (Keep or adapt the testing code as needed, ensuring it uses the new execute_sql)
    print("\nTesting SQL Execution (using psycopg2)...")
    # SELECT example
    sql_result = execute_sql("SELECT id, name, age FROM employees WHERE age > 30 LIMIT 2;")
    print(f"SQL SELECT Result: {sql_result}")
    # INSERT example (make sure 'Test User Psql' doesn't violate constraints)
    # sql_insert_result = execute_sql("INSERT INTO employees (name, age, salary) VALUES ('Test User Psql', 26, 51000);")
    # print(f"SQL INSERT Result: {sql_insert_result}")
    print("\nUtils testing complete.")