import discord
import random
from discord.ext import commands
import discord_bot.memory as memory
import re
from typing import Any, List, Optional, Tuple, Union
from langchain.schema import (AIMessage, BaseMessage, HumanMessage, SystemMessage)
from chatbot.conversation import Chatbot
from discord_bot.state import BotState, empty_active_users, user_exists, new_user
from discord_bot.memory import chat_history, active_users, get_user_chat_history, add_to_chat_history, set_user_active, set_user_inactive
from nlqs.workflow import main_workflow
from openai import chat

# Global variable to store the state of the bot
global_state = BotState.IDLE


def create_bot() -> commands.Bot:
    """Create a Discord Bot

    Returns:
        commands.Bot: The Discord Bot
    """    
    global global_state

    bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
    chatbot_instance = Chatbot.instance()
    
    # Create the default bot behaviors here

    #Event
    @bot.event
    async def on_ready() -> None:
        """Prints a message when the bot is connected to Discord
        """        
        global global_state

        if bot.user is None:
            print("log the relevant error")
            return

        print(f'{bot.user.name} has connected to Discord!') # Prints in the terminal
        print(f'Bot ID: {bot.user.id}') 

        # Save the bot's ID
        memory.bot_id = bot.user.id

        global_state = BotState.IDLE # Initial State - Idle
    


    #Event
    @bot.event
    async def on_message(message) -> None:# Whenever a msg is sent
        """Handles the message sent by the user

        Args:
            message (_type_): The message sent by the user
        """        
        global global_state

        user_input = message.content # Message from the user
        user_id = message.author.id # ID of the user

        # Tracks the last 100 messages in chat_history
        add_to_chat_history(user_id, user_input)

        print(chat_history)
        print(active_users)

        # To prevent bot from replying to it's own message
        if (message.author == bot.user):
            return
        
        if bot.user is None:
            print("log a relevant error")
            return

        # When the bot is mentioned in the message
        if bot.user.mentioned_in(message):
            user_input = remove_user_id(user_input)
            print(f"User Input: {user_input}")
            if user_id in active_users:
                if "!exit" in user_input: # To remove conversation
                    await message.channel.send(f"Conversation with the user <@{user_id}> Ended.")
                    set_user_inactive(user_id) # Removes the user from active_users
                    global_state = empty_active_users(global_state) # Sets the bot state to Idle if active_users are empty
                else: #Any other prompt
                    global_state = user_exists() # Sets the bot state to Engaged
            else:
                set_user_active(user_id) # Adding the user to the current going-on conversations
                global_state = new_user() # Sets the bot state to Engaged

        # To Check the state of the bot
        if global_state == BotState.ENGAGED: # If the bot is in Engaged state (user_conversations exist)
            if user_id in active_users:
                # Assume interaction with the user ......                
                # Set the typing state on the channel
                await message.channel.typing()
                
                queried_data, user_chat_history = main_workflow(user_input, chat_history[:-1])
                print(f"Queried Data: {queried_data}")
                print(f"User Chat History: {user_chat_history}")
                
                corrected_chat_history = change_chat_history(user_chat_history)
                
                if queried_data is None:
                    print("ERROR - Summarization failed")
                    queried_data = ""

                updated_user_input = 'user input: ' + user_input + 'data retrieved for the user input :' + queried_data
                print(f"corrected chat history: {corrected_chat_history}")
                
                print(f"User Input: {user_input}")
                reply = chatbot_instance.converse(user_input=updated_user_input, previous_messages=corrected_chat_history)
                reply = f"<@{user_id}> " + reply[0]
                
                await message.channel.send(reply)
        # To process the commands
        await bot.process_commands(message)


    #Commands

    #BYE
    @bot.command(name='bye', help="-Will end the conversation")
    async def bye(ctx) -> None:
        """Ends the conversation with the user

        Args:
            ctx (Unknown): The context of the command
        """        
        global global_state
        user_id = ctx.author.id
        replies = [f"Goodbye <@{user_id}>! Have a great day!", f"Bye <@{user_id}>! Hope to see you soon!", f"See you later <@{user_id}>!"]
        reply = random.choice(replies)
        await ctx.send(reply)
        empty_active_users(global_state)

    #STATE
    @bot.command(name="state", help="-Prompts the current state of bot")
    async def state(ctx) -> None:
        """Prompts the current state of the bot

        Args:
            ctx (Unknown): The context of the command
        """        
        global global_state

        # Check the state of the bot is it Idle or Engaged and 
        # send the message accordingly
        if global_state == BotState.IDLE:
            await ctx.send("Idle")
        else:
            await ctx.send("Engaged")

    return bot

from langchain.schema import HumanMessage, AIMessage

def change_chat_history(user_chat_history: List[Tuple[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
    """Converts a list of tuples representing chat history to a list of HumanMessage and AIMessage objects.

    Args:
        user_chat_history (List[Tuple[str, str]]): A list of tuples where each tuple represents a message in the chat history.
                                                    The first element of the tuple is the sender (either "Human" or "AI")
                                                    and the second element is the message content.

    Returns:
        List[Union[HumanMessage, AIMessage]]: A list of HumanMessage and AIMessage objects representing the chat history.
    """
    corrected_chat_history = []
    for sender, message in user_chat_history:
        if sender is memory.bot_id:
            corrected_chat_history.append(AIMessage(content=message))
        else:
            corrected_chat_history.append(HumanMessage(content=message))
            
    return corrected_chat_history

def remove_user_id(input_string: str) -> str:
    """Removes user IDs from a string

    Args:
        input_string (str): The input string

    Returns:
        str: The string with user IDs removed
    """

    # Regular expression pattern to match user IDs in the format <@user_id>
    pattern = r"<@\d+>"
    
    # Replace all occurrences of the pattern with an empty string
    result = re.sub(pattern, "", input_string)
    
    # Strip any leading or trailing whitespace
    return result.strip()

