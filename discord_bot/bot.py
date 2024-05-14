import discord
import random
from discord.ext import commands
from chatbot import Chatbot
from discord_bot.state import BotState
from discord_bot.memory import chat_log, user_conversations


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
        global_state = BotState.IDLE


    #Event
    @bot.event
    async def on_message(message):# Whenever a msg is sent
        global global_state

        user_input = message.content

        # Tracks the last 100 messages
        chat_log.append(message.content)
        if len(chat_log) > 100:
            chat_log.pop(0)

        print(chat_log)
        print(user_conversations)
        # To prevent bot from replying to it's own message
        if (message.author == bot.user):
            return
        
        if bot.user.mentioned_in(message):# When the bot is mentioned
            user_id = message.author.id # To tag the user in the reply msg


            if "!bye" in message.content: # To remove conversation
                await message.channel.send(f"Conversation with the user <@{user_id}> Ended.")
                user_conversations.remove(user_id)
                # Updates the bot state to Idle if conversations are empty
                if not user_conversations:
                    global_state = BotState.IDLE
            elif message.content == f"<@{bot.user.id}>":
                user_conversations.append(user_id)
                await message.channel.send(f"New Convo started with the user <@{user_id}>")
                await message.channel.send(f"Hello! How can I help you?")
                global_state = BotState.ENGAGED # Set the state to engaged
            else:
                if user_id in user_conversations: # Conversation already exists with user
                    # Assume interaction with the user ......
                    # TODO: pass the data to the conversational chatbot
                    # TODO: make the below line work
                    reply, _ = chatbot_instance.converse(user_input, {})
                    # reply = f"This is a dummy reply to whatever the user, <@{user_id}> asked!"
                    
                    # reply = "I'm sorry, I'm not sure how to respond to that."
                    await message.channel.send(reply)
                else: # New Conversation with the user
                    user_conversations.append(user_id)
                    # await message.channel.send(f"New Convo started with the user <@{user_id}>")
                    # Assume interaction with the user......
                    # TODO: pass the data to the conversational chatbot
                    # TODO: make the below line work
                    reply, _ = chatbot_instance.converse(user_input, {})
                    # reply = f"This is a dummy reply to whatever the user, <@{user_id}> asked!"
                    global_state = BotState.ENGAGED # Set the state to engaged
                    await message.channel.send(reply) # To send message in the channel
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
        if not user_conversations:
            global_state = BotState.IDLE

    #STATE
    @bot.command(name="state", help="-Prompts the current state of bot")
    async def state(ctx):
        global global_state
        if global_state == BotState.IDLE:
            await ctx.send("Idle")
        else:
            await ctx.send("Engaged")

    return bot