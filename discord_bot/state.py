from enum import Enum

from discord_bot.memory import active_users


class BotState(Enum):
    """A class to represent the state of the bot

    Args:
        Enum (int): The state of the bot
    """

    IDLE = 1
    ENGAGED = 2


def empty_active_users(state_variable: BotState) -> BotState:
    """Sets state to Idle if the active_users is empty

    Args:
        state_variable (BotState): The current state of the bot

    Returns:
        BotState: The new state of the bot
    """
    if state_variable == BotState.ENGAGED:
        if not active_users:  # If active_users is empty
            return BotState.IDLE

    return state_variable


def new_user() -> BotState:
    """Sets state to Engaged when a new Conversation with a user starts

    Returns:
        BotState: The new state of the bot
    """
    return BotState.ENGAGED


def user_exists() -> BotState:
    """Sets state to Engaged when a user already exists in the active_users list

    Returns:
        BotState: The new state of the bot
    """
    return BotState.ENGAGED
