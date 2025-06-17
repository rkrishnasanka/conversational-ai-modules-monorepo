"""
Basic tests for NLQS core functionality.
This file contains minimal tests to verify the main workflow works.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from nlqs.nlqs import NLQS, NLQSResult
from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.vectordb_driver import ChromaDBConfig
from nlqs.summarization import SummarizedInput


def test_nlqs_initialization():
    """Test NLQS class initialization with mock dependencies."""
    
    # Create mock config
    db_config = SQLiteConnectionConfig(
        db_file=Path("test.db"),
        dataset_table_name="test_table",
        uri_column="uri",
        output_columns=["col1", "col2"]
    )
    
    chroma_config = ChromaDBConfig(
        host="localhost",
        port=8000,
        chroma_db_impl="duckdb+parquet",
        persist_directory="test_chroma"
    )
    
    with patch('nlqs.nlqs.SQLiteDriver') as mock_driver, \
         patch('nlqs.nlqs.VectorDBDriver') as mock_vectordb, \
         patch('nlqs.nlqs.get_default_llm') as mock_llm, \
         patch('nlqs.nlqs.OpenAIEmbeddings') as mock_embeddings:
        
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        
        mock_vectordb_instance = Mock()
        mock_vectordb_instance.check_nlqs_collections_exists.return_value = True
        mock_vectordb.return_value = mock_vectordb_instance
        
        # Initialize NLQS
        nlqs = NLQS(db_config, chroma_config)
        
        # Verify initialization
        assert nlqs.table_name == "test_table"
        assert nlqs.uri_column == "uri"
        assert nlqs.output_columns == ["col1", "col2"]
        mock_driver_instance.connect.assert_called_once()


def test_execute_nlqs_query_workflow_empty_input():
    """Test NLQS workflow with empty user input."""
    
    db_config = SQLiteConnectionConfig(
        db_file=Path("test.db"),
        dataset_table_name="test_table"
    )
    
    chroma_config = ChromaDBConfig(
        host="localhost",
        port=8000,
        chroma_db_impl="duckdb+parquet",
        persist_directory="test_chroma"
    )
    
    with patch('nlqs.nlqs.SQLiteDriver') as mock_driver, \
         patch('nlqs.nlqs.VectorDBDriver') as mock_vectordb, \
         patch('nlqs.nlqs.get_default_llm') as mock_llm, \
         patch('nlqs.nlqs.OpenAIEmbeddings') as mock_embeddings:
        
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        
        mock_vectordb_instance = Mock()
        mock_vectordb_instance.check_nlqs_collections_exists.return_value = True
        mock_vectordb.return_value = mock_vectordb_instance
        
        # Initialize NLQS
        nlqs = NLQS(db_config, chroma_config)
        
        # Test empty input
        result = nlqs.execute_nlqs_query_workflow("", [])
        
        assert isinstance(result, NLQSResult)
        assert result.records == []
        assert result.uris == []


def test_execute_nlqs_query_workflow_sql_injection():
    """Test NLQS workflow with SQL injection intent."""
    
    db_config = SQLiteConnectionConfig(
        db_file=Path("test.db"),
        dataset_table_name="test_table"
    )
    
    chroma_config = ChromaDBConfig(
        host="localhost",
        port=8000,
        chroma_db_impl="duckdb+parquet",
        persist_directory="test_chroma"
    )
    
    with patch('nlqs.nlqs.SQLiteDriver') as mock_driver, \
         patch('nlqs.nlqs.VectorDBDriver') as mock_vectordb, \
         patch('nlqs.nlqs.get_default_llm') as mock_llm, \
         patch('nlqs.nlqs.OpenAIEmbeddings') as mock_embeddings, \
         patch('nlqs.nlqs.summarize') as mock_summarize:
        
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        
        mock_vectordb_instance = Mock()
        mock_vectordb_instance.check_nlqs_collections_exists.return_value = True
        mock_vectordb_instance.retrieve_descriptions_and_types_from_db.return_value = {
            "column_descriptions": {"col1": "test column"},
            "numerical_columns": ["col1"],
            "categorical_columns": [],
            "descriptive_columns": []
        }
        mock_vectordb.return_value = mock_vectordb_instance
        
        # Mock summarize to return SQL injection intent
        mock_summarized_input = SummarizedInput(
            summary="malicious input",
            numerical_data={},
            categorical_data={},
            descriptive_data={},
            identifier_data={},
            user_requested_columns=[],
            user_intent="sql_injection"
        )
        mock_summarize.return_value = mock_summarized_input
        
        # Initialize NLQS
        nlqs = NLQS(db_config, chroma_config)
        
        # Test SQL injection input
        result = nlqs.execute_nlqs_query_workflow("DROP TABLE users;", [])
        
        assert isinstance(result, NLQSResult)
        assert result.records == []
        assert result.uris == []
        assert result.is_input_irrelevant == True


if __name__ == "__main__":
    pytest.main([__file__])
