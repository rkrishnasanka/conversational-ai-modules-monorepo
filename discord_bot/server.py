from dotenv import load_dotenv

from discord_bot.bot import create_bot
from discord_bot.parameters import DISCORD_TOKEN


def run_server():
    bot = create_bot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    run_server()
