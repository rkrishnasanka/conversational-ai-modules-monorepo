# main.py
from state_machine.state_machine import SystemConfig, RecommendationSystem

# Example configuration for cannabis recommendations
cannabis_config = SystemConfig(
    name="Cannabis Product Recommendation System",
    description="A system for providing medical cannabis recommendations based on user symptoms and preferences.",
    initial_message="Hi! I'm here to help recommend cannabis products for your needs. What brings you here today?",
    
    states={
        "INITIAL_INQUIRY": {
            "required_info": ["is_medical_query", "main_symptom"],
            "goal": "Understand the user's initial inquiry and identify the main symptom.",
            "generate_queries": False
        },
        "MEDICAL_ASSESSMENT": {
            "required_info": [
                "additional_symptoms",
                "symptom_severity",
                "symptom_duration",
                "previous_treatments",
                "medical_history",
                "lifestyle_factors"
            ],
            "goal": "Gather comprehensive information about the user's health condition and symptoms.",
            "generate_queries": False
        },
        "RECOMMENDATION": {
            "required_info": [
                "suitable_products",
                "usage_guidelines",
                "precautions",
                "expected_effects",
                "user_preferences"
            ],
            "goal": "Generate product recommendations based on user needs.",
            "generate_queries": True
        },
        "CONCLUSION": {
            "required_info": [
                "user_satisfaction",
                "understood_recommendations",
                "remaining_concerns",
                "next_steps"
            ],
            "goal": "Ensure recommendations are understood and provide clear next steps.",
            "generate_queries": False
        }
    },
    
    knowledge_base={
        "product_types": ["Oil", "Vape", "Edible", "Pill", "Flower"],
        "effects": ["Calming", "Uplifting", "Balanced", "Sleep-aid", "Energizing"],
        "onset_times": ["Quick", "Fast", "Medium", "Slow"],
        "durations": ["Short", "Medium", "Long"],
        "use_times": ["Any time", "Daytime", "Nighttime"],
        "common_issues": ["Pain", "Anxiety", "Sleep", "Nausea", "Mood"],
        "strengths": ["Mild", "Moderate", "Strong"]
    }
)

def main():
    print("Starting application")
    try:
        # Initialize the system with the cannabis configuration
        system = RecommendationSystem(cannabis_config)
        
        print("Welcome to the Product Recommendation Assistant!")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    print("Please type something to continue.")
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Thank you for using the Product Recommendation Assistant. Goodbye!")
                    break
                    
                response = system.chat(user_input)
                print(f"\nAssistant: {response.bot_response}")
                
                if response.generated_queries:
                    print("\nGenerated Queries:")
                    for i, query in enumerate(response.generated_queries, 1):
                        print(f"{i}. {query}")
                        
                if response.follow_up_question:
                    print(f"\n{response.follow_up_question}")
                    if response.sample_options:
                        print("\nSample responses:")
                        for i, option in enumerate(response.sample_options, 1):
                            print(f"{i}. {option}")
                            
                if response.conversation_complete:
                    print("\nThank you for using the Product Recommendation Assistant. Take care!")
                    break
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                print("I apologize, but something went wrong. Let's continue our conversation.")
                
    except Exception as e:
        print(f"Application error: {e}")
        print("An error occurred while starting the application. Please check your configuration.")

if __name__ == "__main__":
    main()