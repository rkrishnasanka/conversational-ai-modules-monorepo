import discord
import random
from discord.ext import commands
import re
from chatbot import Chatbot
from discord_bot.state import BotState, empty_active_users, user_exists, new_user
from discord_bot.memory import chat_history, active_users, get_user_chat_history, add_to_chat_history, set_user_active, set_user_inactive
from discord_bot.inventoryquery import summarize


global_state = BotState.IDLE


def create_bot():
    global global_state

    bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
    chatbot_instance = Chatbot.instance()
    
    # Create the default bot behaviors here

    #Event
    @bot.event
    async def on_ready():
        global global_state
        print(f'{bot.user.name} has connected to Discord!') # Prints in the terminal
        global_state = BotState.IDLE # Initial State - Idle


    #Event
    @bot.event
    async def on_message(message):# Whenever a msg is sent
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
                # TODO: pass the data to the conversational chatbot
                #reply = f"This is a dummy reply to whatever the user, <@{user_id}> asked!"
                reply1 = chatbot_instance.converse(user_input, [])
                reply= summarize(user_input, chat_history)
                if reply is None:
                    print("ERROR - Summarization failed")
                    reply = ""
                print(reply[0])
                reply = f"<@{user_id}> " + reply[0]
                await message.channel.send(reply)
        # To process the commands
        await bot.process_commands(message)


    #Commands

    #BYE
    @bot.command(name='bye', help="-Will end the conversation")
    async def bye(ctx):
        global global_state
        user_id = ctx.author.id
        replies = [f"Goodbye <@{user_id}>! Have a great day!", f"Bye <@{user_id}>! Hope to see you soon!", f"See you later <@{user_id}>!"]
        reply = random.choice(replies)
        await ctx.send(reply)
        empty_active_users(global_state)

    #STATE
    @bot.command(name="state", help="-Prompts the current state of bot")
    async def state(ctx):
        global global_state
        if global_state == BotState.IDLE:
            await ctx.send("Idle")
        else:
            await ctx.send("Engaged")

    return bot

def remove_user_id(input_string):
    # Regular expression pattern to match user IDs in the format <@user_id>
    pattern = r"<@\d+>"
    
    # Replace all occurrences of the pattern with an empty string
    result = re.sub(pattern, "", input_string)
    
    # Strip any leading or trailing whitespace
    return result.strip()