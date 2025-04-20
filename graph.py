# graph.py
from typing import Dict
import logging
from langgraph.graph import StateGraph, END
from nodes import (
    AgentState,
    rewrite_prompt_node,
    router_node,
    sql_exec_node,
    image_handler_node,
    audio_handler_node,
    error_handler_node,
    format_final_result
)
import config

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# --- Add Nodes ---
logging.info("Defining graph nodes...")
workflow.add_node("rewriter", rewrite_prompt_node)
workflow.add_node("router", router_node)
workflow.add_node("sql_exec", sql_exec_node)
workflow.add_node("image_handler", image_handler_node)
workflow.add_node("audio_handler", audio_handler_node)
workflow.add_node("error_handler", error_handler_node)
workflow.add_node("final_result_formatter", format_final_result) # Optional formatting node

# --- Define Edges ---

# Start -> Rewriter
workflow.set_entry_point("rewriter")

# Rewriter -> Router
workflow.add_edge("rewriter", "router")

# Router -> Task Handlers (Conditional)
workflow.add_conditional_edges(
    "router",
    lambda state: state.get("route"), # Get the routing decision
    {
        "sql": "sql_exec",
        "image": "image_handler",
        "audio": "audio_handler",
        "error": "final_result_formatter", # Go to formatter if router failed
        "end": "final_result_formatter" # Go to formatter if router decided to end
    }
)

# SQL Executor -> Error Handler or Formatter (Conditional)
def should_retry_sql(state: AgentState) -> str:
    """Determines if SQL execution should be retried or end."""
    error_message = state.get("error_message")
    retry_count = state.get("sql_retry_count", 0)

    if error_message:
        if retry_count < config.MAX_SQL_RETRIES:
            logging.warning(f"SQL error detected. Routing to error handler (Retry {retry_count + 1}/{config.MAX_SQL_RETRIES}).")
            # We actually go to the error_handler node first, which then routes back to sql_exec
            return "error_handler"
        else:
            logging.error(f"SQL error max retries ({config.MAX_SQL_RETRIES}) reached. Ending.")
            # Pass the final error state to the formatter
            return "final_result_formatter"
    else:
        # No error, SQL successful
        logging.info("SQL execution successful. Routing to final formatter.")
        return "final_result_formatter"

workflow.add_conditional_edges(
    "sql_exec",
    should_retry_sql,
    {
        "error_handler": "error_handler",
        "final_result_formatter": "final_result_formatter" # End on success or max retries
    }
)

# Error Handler -> SQL Executor or Formatter (Conditional)
def route_after_error_handling(state: AgentState) -> str:
    """ Routes from error handler back to SQL exec or ends if unfixable."""
    if state.get("route") == "end": # Check if error handler marked as unfixable
        logging.error("Error handler deemed query unfixable. Ending.")
        return "final_result_formatter"
    elif state.get("query"): # Check if a corrected query was generated
        logging.info("Error handler provided corrected query. Routing back to SQL executor.")
        return "sql_exec"
    else:
         # Should not happen if error handler logic is correct, but as fallback:
         logging.error("Error handler finished without a clear next step. Ending.")
         return "final_result_formatter"

workflow.add_conditional_edges(
    "error_handler",
    route_after_error_handling,
    {
        "sql_exec": "sql_exec",
        "final_result_formatter": "final_result_formatter"
    }
)


# Image/Audio Handlers -> SQL Executor or Formatter (Conditional)
def route_from_handler(state: AgentState) -> str:
    """Determines if the handler needs to execute SQL or can end."""
    if state.get("next_node_override") == "sql_exec" and state.get("query"):
        logging.info("Handler generated SQL. Routing to SQL executor.")
        return "sql_exec"
    elif state.get("route") == 'error': # Check if handler itself errored
        logging.error("Handler encountered an error. Routing to final formatter.")
        return "final_result_formatter"
    else:
        logging.info("Handler finished processing. Routing to final formatter.")
        return "final_result_formatter"

workflow.add_conditional_edges("image_handler", route_from_handler, {
    "sql_exec": "sql_exec",
    "final_result_formatter": "final_result_formatter"
})
workflow.add_conditional_edges("audio_handler", route_from_handler, {
    "sql_exec": "sql_exec",
    "final_result_formatter": "final_result_formatter"
})

# Final Formatter -> END
workflow.add_edge("final_result_formatter", END)


# --- Compile the Graph ---
logging.info("Compiling the graph...")
app = workflow.compile()
logging.info("Graph compiled successfully.")


# --- Function to Run the Graph ---
def run_graph(user_prompt: str) -> Dict:
    """Initializes state and runs the graph with the user prompt."""
    initial_state = AgentState(
        original_prompt=user_prompt,
        rewritten_prompt=None,
        route=None,
        query=None,
        image_urls=None,
        image_results=None,
        audio_urls=None,
        audio_results=None,
        final_result=None,
        error_message=None,
        sql_retry_count=0,
        next_node_override=None
    )
    try:
        final_state = app.invoke(initial_state, {"recursion_limit": 15}) # Add recursion limit
        # Return the relevant output part of the final state
        return {
            "final_result": final_state.get('final_result'),
            "error_message": final_state.get('error_message') # Include error if graph ended on error
            }
    except Exception as e:
        logging.error(f"Error invoking graph: {e}", exc_info=True)
        return {"error_message": f"Graph invocation failed: {str(e)}"}