from expert_system.conversation import Chatbot


def test_converse_verbose():
    chatbot = Chatbot.instance()

    user_input = "Tell me about cannabis plants and the differnt kinds of them"

    response, chat_references = chatbot.converse(user_input)

    print("Response:")
    print(response)

    print("References:")
    for chat_reference in chat_references:
        print(chat_reference.title)
        print(chat_reference.context)
        print(chat_reference.ref_url)


def test_converse_concise():
    chatbot = Chatbot.instance()

    user_input = "Tell me about cannabis plants and the differnt kinds of them"

    response, chat_references = chatbot.converse_concise(user_input)

    print("Response:")
    print(response)

    print("References:")
    for chat_reference in chat_references:
        print(chat_reference.title)
        print(chat_reference.context)
        print(chat_reference.ref_url)
