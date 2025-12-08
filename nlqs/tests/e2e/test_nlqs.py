from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig
from nlqs.neondb_driver import NeonDBConfig


def test_nlsq_api(chroma_config, pg_config, setup_postgres_database):
    nlsq = NLQS(
        connection_config=pg_config,
        chroma_config=chroma_config,
    )

    user_input = "suggest me a product that tastes like smores"

    response = nlsq.execute_nlqs_query_workflow(user_input, [])

    print("Response:")
    print(response)


def test_nlqs_with_neondb(neon_config, pg_config, setup_postgres_database, setup_neondb_vectordb):
    """Test NLQS using NeonDB as the vector database backend."""
    nlqs = NLQS(
        connection_config=pg_config,
        chroma_config=neon_config,  # Pass NeonDB config instead of Chroma
    )

    user_input = "suggest me a product that tastes like smores"

    response = nlqs.execute_nlqs_query_workflow(user_input, [])

    print("NeonDB Response:")
    print(response)
    
    # Basic assertions to verify the response structure
    assert hasattr(response, 'records'), "Response should have records attribute"
    assert hasattr(response, 'uris'), "Response should have uris attribute"
    assert isinstance(response.records, list), "Records should be a list"
    assert isinstance(response.uris, list), "URIs should be a list"

