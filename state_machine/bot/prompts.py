def get_json_output_prompt() -> str:
    return """
        User Query: {user_query}
        Final State: {final_state}
        Thought Path: {thought_path}
        Sample Data: {sample_data}

        Generate a JSON object with the following structure:
        {{
            "response": {{
                "text": "The next response or question to the user",
                "type": "question" or "recommendation" or "information",
                "options": ["List", "of", "relevant", "options"],
                "entities": ["List", "of", "key", "entities"],
                "intent": "Identified intent"
                "options": ["List", "of", "relevant", "options"],

            }},
            "recommendation": {{
                "product_type": "Recommended product type",
                "cannabinoid_profile": "Recommended cannabinoid profile",
                "usage_instructions": "Instructions for use",
                "specific_products": ["List", "of", "specific", "product", "recommendations"]
            }},
            "explanation": "Detailed explanation of the response or recommendation",
            "follow_up_questions": ["List", "of", "potential", "follow-up", "questions"],
            "contextual_analysis": "In-depth contextual analysis based on the conversation",
            "relationships": "Identified relationships between entities in the context"
        }}

        If a recommendation is not ready, set the "recommendation" field to null.
        Ensure the response is relevant to the conversation history and helps gather necessary information for a cannabis recommendation or provides appropriate information/recommendations.
        Base your responses and recommendations on the context of cannabis products and user preferences.
        Provide meaningful and diverse options for questions, tailored to the specific query and context.
        Generate a list of potential follow-up questions based on the current context and information gaps.
        Use the provided algorithm framework to guide the questioning process and product recommendations.
        """


def get_classification_prompt() -> str:
    return """
        Classify the user's intent based on the following input:

        User Input: {user_input}

        Possible intents:
        1. Seeking cannabis product recommendation
        2. Providing information about preferences or needs
        3. Inquiring about cannabis effects or usage
        4. Other (not related to cannabis recommendations)

        Respond with only the number corresponding to the intent.
        """


def get_thought_generation_prompt() -> str:
    return """
        Given the current state of the problem:

        {current_state}

        Generate {num_thoughts} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem.

        Your response should be in the following format:
        1. [First thought]
        2. [Second thought]
        ...
        {num_thoughts}. [Last thought]
        
        generate potential follow-up questions, responses, or recommendations in the cannabis recommendation process.
        Use the following sample data structure to guide your questions:

        

        Consider the following aspects based on the sample data columns:
        1. Product Category: Ask about preferred consumption methods (e.g., Tincture, Vaporizer, Edible, Capsule, Flower)
        2. CBD and THC content: Inquire about desired cannabinoid ratios or potency
        3. Onset: Ask about how quickly the user needs the effects to start (e.g., Immediate, Fast, Medium, Slow)
        4. Duration: Inquire about how long they want the effects to last (e.g., Short, Medium, Long)
        5. Effects: Ask about desired effects (e.g., Non-euphoric, Euphoric, Balanced, Sedating, Energizing, Relaxing)
        6. TimeOfUse: When they plan to use the product (e.g., Any, Day, Night)

        Generate thoughts about what information is still needed, what cannabis product recommendations might be appropriate, or what information should be provided to the user based on the conversation so far.
        Ensure that follow-up questions and options are diverse, not repetitive, and tailored to the specific context of the conversation.
        For each question, provide a list of relevant options for the user to choose from, based on the unique values in the sample data.
        
        Make sure you do not repeat the same thought twice.
        """


def get_evaluation_prompt() -> str:
    return """
        Evaluate the potential next steps in the cannabis recommendation conversation based on the following criteria:
        1. Relevance to the user's stated needs and preferences for cannabis products
        2. Alignment with the algorithm framework for cannabis product recommendations
        3. Completeness of information gathered for making a cannabis recommendation
        4. Clarity and helpfulness of the response for the user
        5. Progression towards a suitable product recommendation

        Provide a score from 1-10 for each potential next step, where 10 is the best.
        """


def get_sample_data() -> str:
    # Sample data for guiding recommendations
    return """Product,Category,CBD,THC,Onset,Duration,Effects,TimeOfUse
    CBD Oil Tincture,Tincture,500,10,Fast,Medium,Non-euphoric,Any
    THC Vape Pen,Vaporizer,0,250,Immediate,Short,Euphoric,Any
    1:1 CBD:THC Gummies,Edible,100,100,Slow,Long,Balanced,Any
    CBN Sleep Capsules,Capsule,50,10,Medium,Long,Sedating,Night
    Sativa Flower,Flower,20,180,Immediate,Medium,Energizing,Day
    Indica Flower,Flower,30,220,Immediate,Medium,Relaxing,Night"""
