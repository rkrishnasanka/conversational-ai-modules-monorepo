# **State Machine of Thought**  

## **Simple Explanation**  

### **The State Machine Architecture:**  
- Acts as a **state machine** guiding a conversation through different stages.  
- Each **state** has **specific goals** and collects required information.  
- Tracks what **information is gathered** and what’s still needed.  
- Moves to the **next state** based on collected data.  

### **The Configuration Model:**  
- Serves as a **blueprint** that defines how the system operates.  

## **Overview**  

A flexible recommendation system that guides users through a structured conversation using a state machine approach.  

![State Machine Architecture](./Cannabis_workflow.png) 

## Creating Your Own Recommendation System

### Step 1: Define Your States

Each state represents a conversation phase with specific goals and required information. Here's how to create them:  

```python
from state_machine.state_machine import SystemConfig, RecommendationSystem

# Example: Restaurant Recommendation System
restaurant_config = SystemConfig(
    name="Restaurant Recommendation System",
    description="Helps users find restaurants matching their preferences",
    initial_message="Hello! Looking for a restaurant recommendation?",
    
    states={
        "INITIAL_INQUIRY": {
            "required_info": ["cuisine_type", "location"],
            "goal": "Understand basic dining preferences",
            "generate_queries": False
        },
        "DETAILED_PREFERENCES": {
            "required_info": ["price_range", "dietary_restrictions", "occasion"],
            "goal": "Gather specific requirements for better matching",
            "generate_queries": False
        },
        "RECOMMENDATION": {
            "required_info": ["suitable_restaurants"],
            "goal": "Provide tailored restaurant suggestions",
            "generate_queries": True
        },
        "CONCLUSION": {
            "required_info": ["user_satisfaction", "next_steps"],
            "goal": "Ensure satisfaction and offer reservation help",
            "generate_queries": False
        }
    },
    
    knowledge_base={
        "cuisine_types": ["Italian", "Japanese", "Mexican", "Indian", "American"],
        "price_ranges": ["Budget", "Mid-range", "Upscale", "Luxury"],
        "occasions": ["Date night", "Family dinner", "Business meal", "Casual lunch"]
    }
)
```

### Step 2: Understanding State Components

For each state, define:  

1. **required_info**: List of data points to collect in this state  
2. **goal**: The purpose of this conversation stage  
3. **generate_queries**: Whether to generate search queries (typically for recommendation state)  

### Step 3: Build Your Knowledge Base

The knowledge base contains domain-specific information used for recommendations:  

```python
knowledge_base={
    "category1": ["option1", "option2", "option3"],
    "category2": ["choice1", "choice2", "choice3"]
}
```

### Step 4: Initialize and Run Your System

```python
# Initialize the system
restaurant_system = RecommendationSystem(restaurant_config)

# Example usage in a loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    response = restaurant_system.chat(user_input)
    print(f"System: {response.bot_response}")
    
    if response.follow_up_question:
        print(f"Follow-up: {response.follow_up_question}")
```

## Example Explained: Cannabis Recommendation System

The cannabis example demonstrates a complete implementation:  

### **State Breakdown**  

1. **INITIAL_INQUIRY**:  
   - Collects: Main symptom and whether it's a medical query  
   - Goal: Understand why the user is seeking help  
   - Example: "What brings you here today?"  

2. **MEDICAL_ASSESSMENT**:  
   - Collects: Symptom details, medical history, lifestyle factors  
   - Goal: Build comprehensive understanding of medical needs  
   - Example: "How severe is your pain on a scale of 1-10?"  

3. **RECOMMENDATION**:  
   - Collects: Suitable products, usage guidelines, user preferences  
   - Goal: Generate personalized product recommendations  
   - Example: "Based on your symptoms, I recommend..."  

4. **CONCLUSION**:  
   - Collects: User satisfaction, remaining concerns  
   - Goal: Ensure recommendations are understood  
   - Example: "Do you have any questions about these recommendations?"  

### **Key Design Tips**  

1. **State Order**: Arrange states in a logical conversation flow  
2. **Required Info**: Only include truly necessary information  
3. **Clear Goals**: Each state should have a distinct purpose  
4. **Knowledge Base**: Include all relevant categories for your domain  

<!-- ## **Sample Conversation Flow**  

![Cannabis Workflow](./ToG.png)   -->

## **Future Improvements**  

We're working on enhancing the system with:  

- Auto tuning prompts and system to include rules that ensure boundary conditions are met more clearly preventing unnecessary state transitions.

- **Tree of Thoughts**: For exploring multiple reasoning paths  
  ![Tree of Thoughts](./tree_of_thoughts.jpeg)

- **Knowledge Graph Integration**: For better concept relationships  
  ![Knowledge Graph](./ToG.png)  


