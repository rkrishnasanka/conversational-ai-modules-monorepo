import logging
from pathlib import Path
from state_machine.bot.chat_bot import CannabisRecommendationBot

# Define the log directory and file path
log_dir = Path('./logs')
log_file = log_dir / 'cannabis_bot.log'

# Create the log directory if it does not exist
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def main():
    bot = CannabisRecommendationBot()
    print("Welcome to the Cannabis Recommendation Bot!")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = bot.process_user_input(user_input, chat_history)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
