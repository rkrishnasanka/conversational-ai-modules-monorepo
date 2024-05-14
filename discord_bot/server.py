from discord_bot.bot import create_bot
from dotenv import load_dotenv

TOKEN= 'MTIzNDQ5Njc4NzM0ODQ1NTQ4NQ.GtA7RT.f489dYqYxq0xELxn34pMoD8kIYVeN6DrP6x7uQ'


def run_server():

    bot = create_bot()
    bot.run(TOKEN)


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()
    run_server()
