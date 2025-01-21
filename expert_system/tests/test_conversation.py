from expert_system.conversation import Chatbot


def test_is_conversational_query():

    # Create a new instance of the Conversation class
    chatbot = Chatbot.instance()

    # Test the is_conversational_query method
    is_converational, response = chatbot.is_conversational_query(
        user_input="how are you?")

    assert is_converational is True

    # Test the is_conversational_query method
    is_converational, response = chatbot.is_conversational_query(
        user_input="Explain the consition of cannabanoids?")
    assert is_converational is False
