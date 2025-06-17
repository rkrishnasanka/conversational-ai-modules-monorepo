import uuid
from datetime import datetime
from typing import Any, Dict


def initialize_session_data() -> Dict[str, Any]:
    """
    Initialize the session data structure.

    Returns:
        Dict[str, Any]: Initialized session data.
    """
    return {
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user_context": {"initial_query": {}, "interactions": [], "final_summary": {}},
        "rag_analysis": {"contextual_analysis": {}, "recommendations": {}},
        "cannabis_preferences": {},
    }


def update_session_data(session_data: Dict[str, Any], tot_output: Dict[str, Any]):
    """
    Update session data with new information from Tree of Thoughts output.

    Args:
        session_data (Dict[str, Any]): Current session data.
        tot_output (Dict[str, Any]): Output from the Tree of Thoughts executor.
    """
    response = tot_output["response"]
    session_data["user_context"]["interactions"][-1]["bot_response"] = response

    if response["type"] == "question":
        session_data["cannabis_preferences"].update({entity: True for entity in response.get("entities", [])})

    if response["type"] == "recommendation":
        session_data["rag_analysis"]["recommendations"] = response.get("recommendations", {})
