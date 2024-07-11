from typing import List, Tuple

chat_history: List[Tuple[str, str]] = []  # Stores the tuple of  (user_id, message)
active_users: List[str] = []  # Stores the user_id of active users
bot_id: str = "Replace"  # Bot ID

CHAT_HISTORY_LIMIT = 100


def get_user_chat_history(user_id: str) -> List[str]:
    """Get the chat history of a user

    Args:
        user_id (str): User ID

    Returns:
        List[str]: List of messages
    """
    return [msg for uid, msg in chat_history if uid == user_id]


def add_to_chat_history(user_id: str, message: str) -> None:
    """Add a message to the chat history

    Args:
        user_id (str): User ID
        message (str): Message
    """
    if len(chat_history) >= CHAT_HISTORY_LIMIT:
        chat_history.pop(0)
    chat_history.append((user_id, message))


def set_user_active(user_id: str) -> None:
    """Set the user as active

    Args:
        user_id (str): User ID
    """
    if user_id not in active_users:
        active_users.append(user_id)


def set_user_inactive(user_id: str) -> None:
    """Set the user as inactive

    Args:
        user_id (str): User ID
    """
    if user_id in active_users:
        active_users.remove(user_id)
