# nodes.py
import config
import utils
import logging
import re
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

# --- State Definition ---
class AgentState(TypedDict):
    original_prompt: str
    rewritten_prompt: Optional[str]
    route: Optional[str] # 'sql', 'image', 'audio', 'error', 'end'
    query: Optional[str] # SQL query
    image_urls: Optional[List[str]]
    image_results: Optional[List[Dict[str, Any]]] # Store OCR/analysis results per image
    audio_urls: Optional[List[str]]
    audio_results: Optional[List[Dict[str, Any]]] # Store transcription/summary per audio
    final_result: Optional[Any] # Final text answer or data
    error_message: Optional[str]
    sql_retry_count: int
    # Field to guide handlers towards SQL generation or direct answering
    next_node_override: Optional[str] # e.g., 'sql_exec' or '__end__'

# --- Node Functions ---

def rewrite_prompt_node(state: AgentState) -> AgentState:
    """Rewrites the user prompt for clarity and actionability."""
    logging.info("--- Entering REWRITER NODE ---")
    original_prompt = state['original_prompt']
    system_prompt = (
        "You are an expert prompt engineer. Rewrite the following user request "
        "into a clear, structured command. Focus on the core task "
        "and necessary details. Identify entities, actions, and constraints. "
        "If there is request to add any r"
        "If it involves files, mention them clearly. "
        "Output only the refined prompt. not any query or SQL. "
        "Donot include any explanations and DONOT INCLUDE ANY FALSE OR EXTRA INFORMATION, just the rewritten prompt."
        f"Context about database schema if relevant:\n{config.DB_SCHEMA_INFO}"
        
    )
    rewritten = utils.call_groq_llama(original_prompt, system_prompt)

    if rewritten:
        logging.info(f"Rewritten prompt: {rewritten}")
        state['rewritten_prompt'] = rewritten
    else:
        logging.warning("Failed to rewrite prompt, using original.")
        state['rewritten_prompt'] = original_prompt # Fallback
        state['error_message'] = "Failed to rewrite prompt via LLM."
    return state

def router_node(state: AgentState) -> AgentState:
    """Determines the next step based on the rewritten prompt."""
    logging.info("--- Entering ROUTER NODE ---")
    prompt = state.get('rewritten_prompt') or state.get('original_prompt')
    if not prompt:
        logging.error("Router Error: No prompt available.")
        state['route'] = 'error'
        state['error_message'] = "No prompt provided to router."
        return state

    # # Simple keyword checking first (can be expanded)
    # prompt_lower = prompt.lower()
    # if any(kw in prompt_lower for kw in ['image', 'receipt', '.png', '.jpg', '.jpeg', 'picture']):
    #     logging.info("Routing to IMAGE_HANDLER based on keywords.")
    #     state['route'] = 'image'
    #     return state
    # if any(kw in prompt_lower for kw in ['audio', 'voice', 'summary', '.mp3', '.wav', 'transcribe']):
    #     logging.info("Routing to AUDIO_HANDLER based on keywords.")
    #     state['route'] = 'audio'
    #     return state

    # If no keywords, use LLM for routing
    system_prompt = (
        "Analyze the following user request and determine the primary task type. "
        "Respond with only one word: 'sql', 'image', 'audio', or 'general'. \n"
        "'sql': If the request explicitly asks for database operations (fetch, insert, update, delete, query, data retrieval) or implies interacting with tables like 'employees' or 'refund_requests'.\n"
        "'image': If the request explicitly mentions image files, OCR, reading from pictures, or file names ending in image extensions (.png, .jpg etc).\n"
        "'audio': If the request explicitly mentions audio files, transcription, voice notes, or file names ending in audio extensions (.mp3, .wav etc).\n"
        "'general': If it's a general question or task not fitting the other categories.\n"
         f"Database Schema Context:\n{config.DB_SCHEMA_INFO}"
    )
    route_decision = utils.call_groq_llama(prompt, system_prompt)

    if route_decision:
        route = route_decision.lower().strip().replace("'", "").replace('"', '')
        logging.info(f"LLM route decision: {route}")
        if route in ['sql', 'image', 'audio']:
            state['route'] = route
        elif route == 'general':
            # For now, treat general as potentially needing SQL or direct answer
            # Let's default to trying SQL generation for db-related questions
             logging.info("General query, defaulting to SQL path for potential DB interaction.")
             state['route'] = 'sql' # Or handle general queries differently
        else:
             logging.warning(f"Unknown route decision: {route}. Defaulting to SQL.")
             state['route'] = 'sql' # Fallback
    else:
        logging.error("Failed to get route decision from LLM. Defaulting to SQL.")
        state['route'] = 'sql' # Fallback
        state['error_message'] = "Failed to determine route via LLM."

    logging.info(f"Routing decision: {state['route']}")
    return state

def sql_exec_node(state: AgentState) -> AgentState:
    """Generates (if needed) and executes SQL queries."""
    logging.info("--- Entering SQL_EXEC NODE ---")
    state['error_message'] = None # Clear previous errors for this node
    query = state.get('query')

    if not query:
        # Generate query if not provided (e.g., from error handler or image/audio handler)
        logging.info("No query provided, generating SQL from rewritten prompt...")
        prompt = state.get('rewritten_prompt')
        if not prompt:
            state['error_message'] = "No prompt available to generate SQL query."
            logging.error(state['error_message'])
            state['route'] = 'error' # Signal failure
            return state

        system_prompt = (
            "You are an expert SQL generator. Based on the user request and the database schema, generate a *single*, runnable SQL query for PostgreSQL.\n"
            "IMPORTANT RULES:\n"
            "1. ONLY output the SQL query, with no explanations, comments, or markdown formatting (like ```sql).\n"
            "2. **For INSERT statements: Do NOT include columns that are 'auto-generated' or have database 'defaults' (like 'id', 'created_at') in the column list or VALUES.** Let the database handle them.\n"
            "3. If the user prompt is missing values for required columns that do NOT have defaults (check schema), ask for clarification using 'QUERY_AMBIGUOUS: [Specify missing information]'. Do not guess values for required columns.\n"
            "4. If the request cannot be translated to SQL, return 'QUERY_NOT_SQL'.\n"
            f"Database Schema:\n{config.DB_SCHEMA_INFO}\n\n"
            "User Request:"
        )
        generated_query = utils.call_groq_llama(prompt, system_prompt)

        if not generated_query:
            state['error_message'] = "Failed to generate SQL query via LLM."
            logging.error(state['error_message'])
            state['route'] = 'error'
            return state
        elif generated_query == "QUERY_NOT_SQL":
            state['error_message'] = f"Query generation issue: {generated_query}"
            logging.warning(state['error_message'])
            state['final_result'] = generated_query # Pass clarification back to user
            state['route'] = 'end' # End the flow here
            return state
        else:
            # Basic cleaning: remove markdown backticks if LLM adds them
            query = generated_query.strip().replace('```sql', '').replace('```', '').strip()
            logging.info(f"Generated SQL query: {query}")
            state['query'] = query # Store the generated query

    # Execute the query
    result = utils.execute_sql(query)

    if result.get("error"):
        logging.error(f"SQL Execution Failed (Attempt {state['sql_retry_count'] + 1}/{config.MAX_SQL_RETRIES}): {result['error']}")
        state['error_message'] = result['error']
        # Retry logic is handled by the conditional edge in the graph
    else:
        logging.info("SQL Execution Successful.")
        state['final_result'] = result.get("data", "SQL query executed successfully.")
        state['query'] = None # Clear query after successful execution
        state['sql_retry_count'] = 0 # Reset retries on success
        state['error_message'] = None
        state['route'] = 'end' # Signal success

    return state

def image_handler_node(state: AgentState) -> AgentState:
    """Handles image-related tasks (OCR, analysis)"""
    logging.info("--- Entering IMAGE_HANDLER NODE ---")
    prompt = state.get('rewritten_prompt')
    image_urls = []
    extracted_texts = []
    extracted_data = [] 
    filenames = []
    state['next_node_override'] = None # Reset override

    # 1. Use LLM to extract filenames with better understanding of ranges
    system_prompt = (
        "You are a filename extraction specialist. From the user's text, extract ALL image filenames or references to image files. "
        "If the user mentions a range or pattern like 'file1.png to file10.png' or 'files are named receipt1.jpg through receipt5.jpg', "
        "list ALL individual filenames in the range. "
        "If the user says something like 'refund_req1.png through refund_req10.png', generate the complete list: refund_req1.png, refund_req2.png, etc. "
        "Format your response as a JSON array of strings containing ONLY the filenames. Only output array with no other text or markdown formatting. "
        "Example Ouput: ['receipt1.jpg', 'receipt2.jpg', 'receipt3.jpg'] "
        "If no image files are mentioned, return an empty array []."
    )
    
    filename_extraction = utils.call_groq_llama(prompt, system_prompt)
    
    try:
        import json
        # basic cleaning of the response
        filename_extraction = filename_extraction.strip().replace('```json', '').replace('```', '').strip()
        filename_extraction = filename_extraction[filename_extraction.index('['):filename_extraction.rindex(']')+1]
        print("Filenames from LLM: ",filename_extraction)
        filenames = json.loads(filename_extraction)
    #     if filenames:
    #         logging.info(f"LLM extracted filenames: {filenames}")
    #     else:
    #         # Fallback to regex for basic cases
    #         basic_filenames = re.findall(r'([\w-]+\.(?:png|jpg|jpeg))', prompt, re.IGNORECASE)
            
    #         # Also check for range patterns manually
    #         range_patterns = re.findall(r'(\w+)(\d+)\.(?:png|jpg|jpeg)(?:.*?)(?:to|through|till|until)(?:.*?)(?:\1)(\d+)\.(?:png|jpg|jpeg)', 
    #                                     prompt, re.IGNORECASE)
            
    #         filenames = basic_filenames
    #         for prefix, start, end in range_patterns:
    #             try:
    #                 start_num = int(start)
    #                 end_num = int(end)
    #                 if end_num > start_num and end_num - start_num <= 100:  # Reasonable limit
    #                     for i in range(start_num, end_num + 1):
    #                         range_filename = f"{prefix}{i}.{start.split('.')[-1]}"
    #                         if range_filename not in filenames:
    #                             filenames.append(range_filename)
    #             except ValueError:
    #                 pass
                    
    #         logging.info(f"Regex extraction found filenames: {filenames}")
    except:
        # If LLM response isn't valid JSON, fall back to basic regex
        logging.warning(f"LLM extraction failed, falling back to regex")
        filenames = re.findall(r'([\w-]+\.(?:png|jpg|jpeg))', prompt, re.IGNORECASE)
    
    urls_in_prompt = re.findall(r'(https?://\S+\.(?:png|jpg|jpeg))', prompt, re.IGNORECASE)

    if filenames:
        logging.info(f"Found potential filenames: {filenames}")
        for fname in filenames:
            url = utils.get_public_url(fname)
            if url:
                image_urls.append(url)
            else:
                logging.warning(f"Could not get public URL for filename: {fname}")
    elif urls_in_prompt:
         logging.info(f"Found direct URLs: {urls_in_prompt}")
         image_urls = urls_in_prompt
    else:
         # Maybe list files and let LLM decide? Or require specific naming?
         # Example: Get all PNGs from refund_requests table if not specified
         logging.warning("No specific image files mentioned. Trying to fetch from 'refund_requests' table.")
         sql_fetch_urls = "SELECT image_url FROM refund_requests WHERE image_url IS NOT NULL;"
         result = utils.execute_sql(sql_fetch_urls)
         if result.get("data") and isinstance(result["data"], list):
             for row in result["data"]:
                 if isinstance(row, dict) and row.get('image_url'):
                     image_urls.append(row['image_url'])
             logging.info(f"Fetched {len(image_urls)} image URLs from table.")
         else:
             state['error_message'] = "Could not find image filenames in prompt or fetch from refund_requests table."
             logging.error(state['error_message'])
             state['route'] = 'error'
             return state

    if not image_urls:
         state['error_message'] = "No image URLs could be determined for processing."
         logging.error(state['error_message'])
         state['route'] = 'error'
         return state

    state['image_urls'] = image_urls
    state['image_results'] = []

    # 2. Process each image with Gemini Vision using the specific prompt
    vision_prompt = "Extract the text from given receipt image and return a json object. Don't add any false information, any extra comments or markdown formatting (like ```json)"

    for url in image_urls:
        logging.info(f"Processing image with Gemini Vision: {url}")
        # Call the new Gemini vision function
        gemini_result_json_str = utils.call_gemini_vision(url, vision_prompt)

        if gemini_result_json_str and not gemini_result_json_str.startswith("Error:"):
            logging.info(f"Gemini Vision result for {url}: {gemini_result_json_str[:150]}...") # Log snippet
            state['image_results'].append({"url": url, "data_json": gemini_result_json_str}) # Store JSON string
            # Prepare data for final LLM call
            extracted_data.append(f"Image ({url}):\n{gemini_result_json_str}")
        elif gemini_result_json_str and gemini_result_json_str.startswith("Error:"):
             # Handle errors reported by the Gemini function (e.g., safety blocks, API key)
             logging.error(f"Gemini Vision failed for {url}: {gemini_result_json_str}")
             state['image_results'].append({"url": url, "data_json": None, "error": gemini_result_json_str})
             # Decide if one error should stop the whole process or just skip the image
             # For now, we'll let it continue and the final LLM will see the errors
             extracted_data.append(f"Image ({url}):\nERROR: {gemini_result_json_str}")
        else:
            # Handle cases where None was returned (e.g., download failed, other exception)
            logging.warning(f"Failed to get valid result from Gemini Vision for {url}")
            state['image_results'].append({"url": url, "data_json": None, "error": "Gemini processing failed (None returned)"})
            extracted_data.append(f"Image ({url}):\nERROR: Processing failed.")

    # 3. Use LLM to interpret results based on original prompt
    if not extracted_data: # Check if any data (even errors) was generated
        state['error_message'] = "No data could be extracted or processed from any of the images using Gemini Vision."
        logging.error(state['error_message'])
        state['route'] = 'error'
        return state

    combined_results = "\n\n".join(extracted_data)
    system_prompt = (
        "You are an AI assistant analyzing JSON data extracted from one or more receipt images using Google Gemini. "
        "Based on the original user request and the extracted JSON data (or error messages) provided below, perform the requested task.\n"
        "Carefully parse the JSON for each image to find relevant information (like total amounts, items, dates etc.). Handle potential errors reported for specific images.\n"
        "If the user wants specific information, extract and present it clearly from the JSON data.\n"
        "If the user wants to UPDATE database records based on this info (e.g., update 'refund_requests' table with amounts from receipts), formulate the necessary SQL UPDATE query/queries. Use the image URL or derive the record ID if possible to match the JSON data to the correct database row. ONLY output the SQL query if that's the task.\n"
        "If multiple rows in the same table need to be updated, generate a **single SQL query** using `CASE WHEN` syntax rather than multiple separate `UPDATE` statements."
        "If generating SQL, use the format: SQL_QUERY: <your sql query>\n"
        "Otherwise, just provide the answer to the user's request based on the analyzed JSON data.\n"
        f"Database Schema Context:\n{config.DB_SCHEMA_INFO}\n\n"
        f"Original User Request: {prompt}\n\n"
        f"Extracted JSON Data/Errors from Image(s):\n{combined_results}" # Now contains JSON strings or errors
    )


    final_analysis = utils.call_groq_llama("Analyze the extracted text based on my original request.", system_prompt)

    if final_analysis:
        if "SQL_QUERY:" in final_analysis:
            # Extract the actual SQL query part after the keyword
            sql_query = final_analysis.split("SQL_QUERY:", 1)[1].strip()
            
            # Optional: normalize or clean up trailing semicolons or extra whitespace
            sql_query = sql_query.strip().rstrip(';') + ';'  # ensure a single semicolon at the end
            
            logging.info("Image handler generated SQL query.")
            state['query'] = sql_query
            state['next_node_override'] = 'sql_exec' # Route to SQL execution
        else:
            logging.info("Image handler generated final textual result.")
            state['final_result'] = final_analysis
            state['next_node_override'] = '__end__' # End the process
    else:
        state['error_message'] = "LLM failed to analyze extracted image text."
        logging.error(state['error_message'])
        state['route'] = 'error'

    return state


def audio_handler_node(state: AgentState) -> AgentState:
    """Handles audio-related tasks (transcription, summarization)"""
    logging.info("--- Entering AUDIO_HANDLER NODE ---")
    prompt = state.get('rewritten_prompt')
    audio_urls = []
    transcriptions = []
    state['next_node_override'] = None  # Reset override

    # 1. Generate SQL query using LLM based on the rewritten prompt
    logging.info("Generating SQL query to fetch audio URLs using LLM...")
    system_prompt = (
        "You are an expert SQL generator. Based on the user request and the database schema, generate a SQL query "
        "to fetch audio URLs relevant to the task described in the prompt. "
        "IMPORTANT: ONLY output the SQL query, with no explanations or markdown formatting (like ```sql). \n"
        f"Database Schema:\n{config.DB_SCHEMA_INFO}\n\n"
        f"User Request: {prompt}"
    )
    generated_query = utils.call_groq_llama(prompt, system_prompt)

    if not generated_query:
        state['error_message'] = "Failed to generate SQL query via LLM."
        logging.error(state['error_message'])
        state['route'] = 'error'
        return state

    # Clean the generated query
    sql_query = generated_query.strip().replace('```sql', '').replace('```', '').strip()
    logging.info(f"Generated SQL query: {sql_query}")

    # 2. Execute the generated SQL query to fetch audio URLs
    logging.info("Executing SQL query to fetch audio URLs...")
    result = utils.execute_sql(sql_query)

    if result.get("data") and isinstance(result["data"], list):
        for row in result["data"]:
            if isinstance(row, dict) and row.get('audio_url'):
                # Store ID along with URL if fetched from DB for potential later use
                audio_urls.append({"id": row.get('id'), "url": row['audio_url']})
        logging.info(f"Fetched {len(audio_urls)} audio URLs from the database.")
    else:
        state['error_message'] = "Could not fetch audio URLs from the database."
        logging.error(state['error_message'])
        state['route'] = 'error'
        return state

    if not audio_urls:
        state['error_message'] = "No audio URLs could be determined for processing."
        logging.error(state['error_message'])
        state['route'] = 'error'
        return state

    # Ensure audio_urls contains only the URL strings for processing
    state['audio_urls'] = [item['url'] if isinstance(item, dict) else item for item in audio_urls]
    state['audio_results'] = []


    # 2. Transcribe each audio file using Groq Whisper
    for item in audio_urls: # Iterate through original list (might contain dicts)
        url = item['url'] if isinstance(item, dict) else item
        record_id = item.get('id') if isinstance(item, dict) else None # Keep track of ID if available
        logging.info(f"Transcribing audio: {url}")
        # Assuming Whisper output might be Urdu, although Whisper Large v3 is multilingual
        transcript = utils.call_groq_whisper(url) # Consider adding language hint: language='ur'

        if transcript:
            logging.info(f"Transcription result for {url}: {transcript[:100]}...")
            # 3. Translate if needed (Example: Assume Urdu -> English)
            # You might want to detect language first or rely on user prompt specifics
            translated_text = transcript
            if "urdu" in prompt.lower() or True: # Simple check, could be more robust lang detection
                logging.info("Attempting translation to English...")
                translated = utils.translate_text(transcript, target_language="English")
                if translated:
                    translated_text = translated
                    logging.info(f"Translated text: {translated_text[:100]}...")
                else:
                    logging.warning(f"Translation failed for {url}, using original transcript.")

            state['audio_results'].append({"url": url, "id": record_id, "transcript": transcript, "processed_text": translated_text})
            transcriptions.append(f"Audio ({url}, ID: {record_id}):\n{translated_text}")
        else:
            logging.warning(f"Transcription failed for {url}")
            state['audio_results'].append({"url": url, "id": record_id, "transcript": None, "processed_text": None, "error": "Transcription failed"})


    # 4. Use LLM to interpret results based on original prompt
    if not transcriptions:
        state['error_message'] = "No text could be transcribed or processed from any audio files."
        logging.error(state['error_message'])
        state['route'] = 'error'
        return state

    combined_transcripts = "\n\n".join(transcriptions)
    system_prompt = (
        "You are an AI assistant analyzing transcribed text from one or more audio files. "
        "Based on the original user request and the processed text provided below, perform the requested task.\n"
        "If the user wants a summary or specific information from the audio, provide it clearly in detail, and ignore retrieving/fetchin part.\n"
        "If the user wants to UPDATE database records based on this info, formulate the necessary SQL UPDATE query/queries. Use the record ID associated with the audio if available. ONLY output the SQL query if that's the task.\n"
        "If generating SQL, use the format: SQL_QUERY: <your sql query>\n"
        "Otherwise, just provide the answer to the user's request based on the text from audio in detail.\n"
        f"Database Schema Context:\n{config.DB_SCHEMA_INFO}\n\n"
        f"Original User Request: {prompt}\n\n"
        f"Processed Text from Audio(s):\n{combined_transcripts}"
    )

    final_analysis = utils.call_groq_llama("Analyze the processed audio text based on my original request.", system_prompt)

    if final_analysis:
        if final_analysis.strip().startswith("SQL_QUERY:"):
            sql_query = final_analysis.replace("SQL_QUERY:", "").strip()
            logging.info("Audio handler generated SQL query.")
            state['query'] = sql_query
            state['next_node_override'] = 'sql_exec' # Route to SQL
        else:
            logging.info("Audio handler generated final textual result.")
            state['final_result'] = final_analysis
            state['next_node_override'] = '__end__' # End the process
    else:
        state['error_message'] = "LLM failed to analyze processed audio text."
        logging.error(state['error_message'])
        state['route'] = 'error'

    return state

def error_handler_node(state: AgentState) -> AgentState:
    """Attempts to fix a failed SQL query using LLM."""
    logging.info("--- Entering ERROR_HANDLER NODE ---")
    original_query = state.get('query')
    error_msg = state.get('error_message')
    prompt = state.get('rewritten_prompt')

    if not original_query or not error_msg:
        state['error_message'] = "Error handler called without query or error message."
        logging.error(state['error_message'])
        state['final_result'] = "An unexpected error occurred in the error handler."
        state['route'] = 'end' # Cannot proceed
        return state

    logging.warning(f"Attempting to fix SQL query. Attempt {state['sql_retry_count'] + 1}")

    system_prompt = (
        "You are an expert SQL debugger. The following PostgreSQL query failed. Analyze the original user request, the failed query, the error message, and the database schema. Generate a *corrected* SQL query.\n"
        "IMPORTANT RULES:\n"
        "1. ONLY output the corrected SQL query, with no explanations or markdown.\n"
        "2. Pay close attention to errors like 'violates not-null constraint', especially on columns like 'id' or 'created_at'. These are often auto-generated.\n"
        "3. **If the failed query is an INSERT statement trying to specify 'auto-generated' columns (like 'id', 'created_at'), correct the query by REMOVING these columns from the column list and VALUES.**\n"
        "4. If the error indicates a missing value for a required column (violates not-null constraint on a column the user *must* provide, like potentially 'age' or 'name' if required), indicate that the query is unfixable without more information using 'QUERY_UNFIXABLE: Missing required value for column [column_name]'.\n"
        "5. If the error is genuinely unfixable or unclear, return 'QUERY_UNFIXABLE: [Reason]'.\n"
         f"Database Schema:\n{config.DB_SCHEMA_INFO}\n\n"
         f"Original User Request: {prompt}\n"
         f"Failed SQL Query:\n{original_query}\n\n"
         f"Error Message:\n{error_msg}\n\n"
         "Corrected SQL Query:"
    )

    corrected_query_response = utils.call_groq_llama("Generate the corrected query.", system_prompt)

    if not corrected_query_response:
        state['error_message'] = f"LLM failed to generate a corrected query. Original error: {error_msg}"
        logging.error(state['error_message'])
        # Keep route as 'error' - the conditional edge will check retries
    elif corrected_query_response.startswith("QUERY_UNFIXABLE"):
        state['error_message'] = f"Query deemed unfixable by LLM: {corrected_query_response}. Original error: {error_msg}"
        logging.error(state['error_message'])
        state['final_result'] = f"Failed to execute and fix the SQL query after retries. Error: {corrected_query_response}"
        state['route'] = 'end' # Stop retrying if LLM says it's unfixable
    else:
        # Basic cleaning
        corrected_query = corrected_query_response.strip().replace('```sql', '').replace('```', '').strip()
        logging.info(f"Generated corrected query: {corrected_query}")
        state['query'] = corrected_query # Update state with the new query
        state['error_message'] = None # Clear error message for the next attempt
        state['sql_retry_count'] += 1 # Increment retry count (checked in edge)
        state['route'] = 'sql_exec' # Explicitly route back to sql_exec

    return state

# --- Helper function for final result formatting (optional) ---
def format_final_result(state: AgentState) -> AgentState:
    """ Cleans up or formats the final result before ending."""
    logging.info("--- Formatting Final Result ---")
    result = state.get('final_result')
    error = state.get('error_message')

    if error:
        logging.error(f"Ending with error: {error}")
        # Ensure critical errors are reported clearly
        if state.get('sql_retry_count', 0) >= config.MAX_SQL_RETRIES:
            state['final_result'] = f"Failed to execute SQL query after {config.MAX_SQL_RETRIES} attempts. Last error: {error}"
        else:
             state['final_result'] = f"An error occurred: {error}"
    elif result is None:
        state['final_result'] = "Task completed, but no specific result was generated."
    # elif isinstance(result, list) and len(result) > 5: # Truncate long lists
    #      state['final_result'] = {"data": result[:5], "message": f"Showing first 5 of {len(result)} results."}
    # Add more formatting rules as needed

    logging.info(f"Final Output: {state['final_result']}")
    return state