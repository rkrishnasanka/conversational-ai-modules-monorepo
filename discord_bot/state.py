from enum import Enum
from memory import active_users


class BotState(Enum):
    IDLE = 1
    ENGAGED = 2

# Sets state to Idle if the active_users is empty
def empty_active_users(state_variable: BotState) -> BotState:
    if state_variable == BotState.ENGAGED:
        if not active_users: # If active_users is empty
            return BotState.IDLE

# Sets state to Engaged when a new Conversation with a user starts
def new_user() -> BotState:
    return BotState.ENGAGED

# Sets state to Engaged when conversation with user already exists
def user_exists() -> BotState:
    return BotState.ENGAGED
