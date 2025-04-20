# main.py
import logging
from graph import run_graph
import sys

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def main():
    """Runs the agent in a loop, taking user input."""
    print("\n--- LangGraph Agent System ---")
    print("Enter your prompts below. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nPrompt> ")
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting agent system.")
                break
            if not user_input:
                continue

            logging.info(f"Received prompt: {user_input}")
            print("Processing...")

            # Run the graph
            result = run_graph(user_input)

            # Print the result
            print("\n--- Agent Result ---")
            if result.get("error_message"):
                print(f"Error: {result['error_message']}")
            elif result.get("final_result"):
                 # Pretty print if it's complex data (like list/dict)
                 if isinstance(result['final_result'], (dict, list)):
                      import json
                      print(json.dumps(result['final_result'], indent=2))
                 else:
                     print(result['final_result'])
            else:
                print("Agent finished, but no specific result was provided.")
            print("--------------------\n")

        except KeyboardInterrupt:
            print("\nExiting agent system.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print(f"An critical error occurred: {e}")

if __name__ == "__main__":
    # Validate Supabase/API clients before starting (basic check)
    try:
        # Import utils here to ensure config is loaded first
        import utils
        if not utils.supabase_storage_client or not utils.groq_client or not utils.whisper_client:
             print("ERROR: One or more clients (Supabase, Groq or Whisper) failed to initialize.")
             print("Please check your .env file and API keys.")
             sys.exit(1)
    except ImportError as e:
         print(f"ERROR: Failed to import utils or config: {e}")
         print("Ensure all files are present and requirements are installed.")
         sys.exit(1)
    except Exception as e:
        print(f"ERROR during initial client check: {e}")
        sys.exit(1)

    main()