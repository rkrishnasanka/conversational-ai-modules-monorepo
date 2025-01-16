from utils.llm import get_default_embedding_function, get_default_llm


def test_get_default_llm():
    # Test the default llm to work
    llm = get_default_llm()

    assert llm is not None

    output = llm.invoke("Tell me a joke")
    assert output is not None


def test_get_default_embeddings():
    # Test the default embeddings to work
    embedding_function = get_default_embedding_function()

    assert embedding_function is not None

    output = embedding_function.embed_query("Tell me a joke")
    assert output is not None
