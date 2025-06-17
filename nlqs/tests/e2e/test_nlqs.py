from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig


def test_nlsq_api(chroma_config, pg_config, setup_postgres_database):
    nlsq = NLQS(
        connection_config=pg_config,
        chroma_config=chroma_config,
    )

    user_input = "suggest me a product that tastes like smores"

    response = nlsq.execute_nlqs_query_workflow(user_input, [])

    print("Response:")
    print(response)
