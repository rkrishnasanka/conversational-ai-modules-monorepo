"""
NeonDB Vector Database Driver for NLQS

This module provides a PostgreSQL/NeonDB backend for NLQS vector operations using pgvector.
It mirrors the VectorDBDriver API but uses PostgreSQL tables instead of ChromaDB collections.

Key differences from VectorDBDriver (ChromaDB):
1. Connection Management: Uses psycopg3 connections instead of ChromaDB client
2. Population Methods: Takes list of dict records instead of pandas DataFrames
3. Not Implemented: qualitative_table_name_search, qualitative_db_name_search, store_column_info_in_db
4. Context Manager: Supports 'with' statement for automatic connection cleanup

Usage:
    config = NeonDBConfig(conn_string="postgresql://...")
    with NeonVectorDBDriver(config, embedding_fn) as driver:
        driver.populate_column_info(records)
        results = driver.get_closest_data_from_description(...)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import psycopg
from psycopg.rows import dict_row

from nlqs.vectordb_driver import (
    ColumnType,
    ColumnDescriptions,
    ClosestDataResult,
    QualitativeSearchResult,
)


@dataclass
class NeonDBConfig:
    # Connection string: postgresql://user:password@host:port/database
    conn_string: str = ""
    schema: str = "public"
    column_info_table: str = "nlqs_column_info"          # mirrors nlqs_column_info collection
    dataset_table: str = "nlqs_descriptive_data"         # mirrors nlqs_descriptive_data collection
    table_desc_table: str = "nlqs_table_descriptions"    # mirrors nlqs_table_descriptions collection
    embedding_dim: int = 1536                            # match your embedding function dimension
    
    def __post_init__(self):
        """Load connection string from environment if not provided."""
        if not self.conn_string:
            self.conn_string = os.getenv("NEONDB_CONNECTION_STRING", "")


def _to_vector_literal(embedding: List[float]) -> str:
    """Convert a list of floats to pgvector literal format.
    
    Args:
        embedding: List of float values
        
    Returns:
        String in pgvector format: '[0.1,0.2,0.3]'
        
    Raises:
        ValueError: If embedding is empty or contains invalid values
    """
    if not embedding:
        raise ValueError("Embedding list cannot be empty")
    
    # Check for invalid values (NaN, Infinity)
    import math
    for i, x in enumerate(embedding):
        if not isinstance(x, (int, float)):
            raise ValueError(f"Embedding value at index {i} must be numeric, got {type(x)}")
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"Embedding contains invalid value at index {i}: {x}")
    
    # pgvector accepts literals like: '[0.1,0.2,0.3]'
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


class NeonVectorDBDriver:
    def __init__(self, neon_config: NeonDBConfig, embedding_function: Callable[[str], List[float]]):
        if not neon_config.conn_string:
            raise ValueError(
                "NEONDB_CONNECTION_STRING is not set. "
                "Please set it in your environment or pass it to NeonDBConfig(conn_string='...')"
            )
        
        # Validate connection string format
        if not neon_config.conn_string.startswith(("postgres://", "postgresql://")):
            raise ValueError(
                f"Invalid connection string format. Expected 'postgresql://...' or 'postgres://...', "
                f"got: {neon_config.conn_string[:20]}..."
            )

        self.config = neon_config
        self.embedding_function = embedding_function
        
        # Keep a single connection; psycopg v3 is thread-safe to create connections as needed
        try:
            self._conn = psycopg.connect(self.config.conn_string)
            self._conn.autocommit = True
        except psycopg.Error as e:
            raise ConnectionError(f"Failed to connect to NeonDB: {e}")

        # Ensure schema/tables exist (safe to call multiple times)
        self.initialize_nlqs_vectordb(self.config)

    def close(self):
        """Close the database connection."""
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False

    # ---------- Schema management ----------
    def check_nlqs_collections_exists(self) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = ANY(%s)
                """,
                (
                    self.config.schema,
                    [self.config.column_info_table, self.config.dataset_table, self.config.table_desc_table],
                ),
            )
            row = cur.fetchone()
            return (row["cnt"] or 0) == 3

    @staticmethod
    def initialize_nlqs_vectordb(config: NeonDBConfig) -> None:
        with psycopg.connect(config.conn_string) as conn:
            conn.autocommit = True
            with conn.cursor(row_factory=dict_row) as cur:
                cur = cast(Any, cur)
                # Enable pgvector
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create schema if it doesn't exist (for non-public schemas)
                if config.schema != "public":
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {config.schema};")

                # Column info table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {config.schema}.{config.column_info_table} (
                        id BIGSERIAL PRIMARY KEY,
                        db_name TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        column_name TEXT NOT NULL,
                        column_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        embedding VECTOR({config.embedding_dim}) NOT NULL
                    );
                    """
                )
                # Descriptive dataset table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {config.schema}.{config.dataset_table} (
                        id BIGSERIAL PRIMARY KEY,
                        db_name TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        column_name TEXT NOT NULL,
                        lookup_key_column_name TEXT NOT NULL,
                        lookup_key_column_value TEXT NOT NULL,
                        description TEXT NOT NULL,
                        embedding VECTOR({config.embedding_dim}) NOT NULL
                    );
                    """
                )
                # Table descriptions table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {config.schema}.{config.table_desc_table} (
                        id BIGSERIAL PRIMARY KEY,
                        db_name TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        embedding VECTOR({config.embedding_dim}) NOT NULL
                    );
                    """
                )

                # Recommended IVF indexes for cosine distance (adjust lists as needed)
                # Note: IVFFlat indexes require data to be present first. If tables are empty,
                # indexes can be created later after data population.
                try:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_colinfo_embedding
                        ON {config.schema}.{config.column_info_table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_dataset_embedding
                        ON {config.schema}.{config.dataset_table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_tabledesc_embedding
                        ON {config.schema}.{config.table_desc_table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    )
                except psycopg.Error as e:
                    # Index creation may require ANALYZE/populated data; not fatal
                    # Common error: "ivfflat index build requires a non-empty table"
                    import warnings
                    warnings.warn(
                        f"Could not create IVFFlat indexes (may require populated tables): {e}. "
                        "This is not critical - indexes can be created manually later if needed."
                    )

    @staticmethod
    def purge_nlqs_vectordb(config: NeonDBConfig) -> None:
        with psycopg.connect(config.conn_string) as conn:
            conn.autocommit = True
            with conn.cursor(row_factory=dict_row) as cur:
                cur = cast(Any, cur)
                cur.execute(f"DROP TABLE IF EXISTS {config.schema}.{config.column_info_table} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {config.schema}.{config.dataset_table} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {config.schema}.{config.table_desc_table} CASCADE;")

    # ---------- Data access APIs (mirror VectorDBDriver) ----------
    def retrieve_descriptions_and_types_from_db(
        self, db_name_filter: Optional[str] = None, table_name_filter: Optional[str] = None
    ) -> Optional[ColumnDescriptions]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            where_clauses = []
            params: List[Any] = []
            if db_name_filter:
                where_clauses.append("db_name = %s")
                params.append(db_name_filter)
            if table_name_filter:
                where_clauses.append("table_name = %s")
                params.append(table_name_filter)
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cur.execute(
                f"""
                SELECT column_name, description, column_type
                FROM {self.config.schema}.{self.config.column_info_table}
                {where_sql}
                """,
                params,
            )
            rows = cur.fetchall()
            if not rows:
                return None

            ret: ColumnDescriptions = {
                "column_descriptions": {},
                "numerical_columns": [],
                "categorical_columns": [],
                "descriptive_columns": [],
                "identifier_columns": [],
            }

            for r in rows:
                col_name = str(r["column_name"])
                desc = str(r["description"])
                col_type = ColumnType(str(r["column_type"]))
                ret["column_descriptions"][col_name] = desc
                if col_type == ColumnType.NUMERICAL:
                    ret["numerical_columns"].append(col_name)
                elif col_type == ColumnType.CATEGORICAL:
                    ret["categorical_columns"].append(col_name)
                elif col_type == ColumnType.DESCRIPTIVE:
                    ret["descriptive_columns"].append(col_name)
                elif col_type == ColumnType.IDENTIFIER:
                    ret["identifier_columns"].append(col_name)

            return ret

    def check_if_column_name_exists(self, column_name: str, table_name: str, db_name: str) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            cur.execute(
                f"""
                SELECT 1
                FROM {self.config.schema}.{self.config.column_info_table}
                WHERE column_name = %s AND table_name = %s AND db_name = %s
                LIMIT 1
                """,
                (column_name, table_name, db_name),
            )
            return cur.fetchone() is not None

    def get_column_type(self, column_name: str, table_name: str, db_name: str) -> ColumnType:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            cur.execute(
                f"""
                SELECT column_type
                FROM {self.config.schema}.{self.config.column_info_table}
                WHERE column_name = %s AND table_name = %s AND db_name = %s
                LIMIT 1
                """,
                (column_name, table_name, db_name),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Column {column_name} not found in the database.")
            return ColumnType(str(row["column_type"]))

    def get_closest_column_from_description(
        self,
        approximate_column_name: str,
        users_description: str,
        sample_data_strings: List[str],
        database_name: str,
        table_name: str,
    ) -> Tuple[str, ColumnType]:
        # Build a query string to embed
        query_text = " | ".join(filter(None, [approximate_column_name, users_description, *sample_data_strings]))
        emb = self.embedding_function(query_text)
        emb_lit = _to_vector_literal(emb)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            cur.execute(
                f"""
                SELECT column_name, column_type
                FROM {self.config.schema}.{self.config.column_info_table}
                WHERE db_name = %s AND table_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (database_name, table_name, emb_lit),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("No metadata found for closest column search.")
            return str(row["column_name"]), ColumnType(str(row["column_type"]))

    def get_closest_data_from_description(
        self,
        column_name: str,
        description: str,
        database_name: str,
        table_name: str,
    ) -> List[ClosestDataResult]:
        emb = self.embedding_function(description)
        emb_lit = _to_vector_literal(emb)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            cur.execute(
                f"""
                SELECT lookup_key_column_name, lookup_key_column_value, description
                FROM {self.config.schema}.{self.config.dataset_table}
                WHERE db_name = %s AND table_name = %s AND column_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 5
                """,
                (database_name, table_name, column_name, emb_lit),
            )
            rows = cur.fetchall() or []

        ret: List[ClosestDataResult] = []
        for r in rows:
            lookup_key = str(r["lookup_key_column_name"])
            column_value = r["lookup_key_column_value"]
            # Try to preserve int type if possible, otherwise convert to string
            if isinstance(column_value, int):
                col_val = column_value
            else:
                try:
                    col_val = int(column_value)
                except (ValueError, TypeError):
                    col_val = str(column_value)
            
            ret.append(
                {
                    "lookup_key": lookup_key,
                    "column_value": col_val,
                    "data": str(r["description"]),
                }
            )
        return ret

    def qualitative_dataset_search(
        self, data: Dict[str, str], table_name: str, db_name: str
    ) -> QualitativeSearchResult:
        # Column-wise similarity search; return primary-key (lookup) candidates
        ids_per_column: QualitativeSearchResult = {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            for column_name, condition in data.items():
                emb = self.embedding_function(str(condition))
                emb_lit = _to_vector_literal(emb)
                cur.execute(
                    f"""
                    SELECT lookup_key_column_name, lookup_key_column_value
                    FROM {self.config.schema}.{self.config.dataset_table}
                    WHERE db_name = %s AND table_name = %s AND column_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5
                    """,
                    (db_name, table_name, column_name, emb_lit),
                )
                rows = cur.fetchall() or []
                if rows:
                    ids_for_column = set(
                        (str(r["lookup_key_column_name"]), str(r["lookup_key_column_value"])) for r in rows
                    )
                    ids_per_column[column_name] = list(ids_for_column)
        return ids_per_column

    # ---------- Bulk population helpers (optional; mirror populate_* in VectorDBDriver) ----------
    def populate_column_info(
        self,
        records: List[Dict[str, Any]],  # keys: db_name, table_name, column_name, column_type, description, embedding(list[float])
        batch_size: int = 100,
    ) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # Validate embedding dimensions
                for idx, r in enumerate(batch):
                    if len(r["embedding"]) != self.config.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch at record {i + idx}: "
                            f"expected {self.config.embedding_dim}, got {len(r['embedding'])}"
                        )
                
                cur.executemany(
                    f"""
                    INSERT INTO {self.config.schema}.{self.config.column_info_table}
                        (db_name, table_name, column_name, column_type, description, embedding)
                    VALUES (%(db_name)s, %(table_name)s, %(column_name)s, %(column_type)s, %(description)s, %(embedding)s::vector)
                    """,
                    [
                        {
                            **r,
                            "embedding": _to_vector_literal(r["embedding"]),
                        }
                        for r in batch
                    ],
                )

    def populate_dataset_info(
        self,
        records: List[Dict[str, Any]],  # keys: db_name, table_name, column_name, lookup_key_column_name, lookup_key_column_value, description, embedding(list[float])
        batch_size: int = 100,
    ) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # Validate embedding dimensions
                for idx, r in enumerate(batch):
                    if len(r["embedding"]) != self.config.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch at record {i + idx}: "
                            f"expected {self.config.embedding_dim}, got {len(r['embedding'])}"
                        )
                
                cur.executemany(
                    f"""
                    INSERT INTO {self.config.schema}.{self.config.dataset_table}
                        (db_name, table_name, column_name, lookup_key_column_name, lookup_key_column_value, description, embedding)
                    VALUES (%(db_name)s, %(table_name)s, %(column_name)s, %(lookup_key_column_name)s, %(lookup_key_column_value)s, %(description)s, %(embedding)s::vector)
                    """,
                    [
                        {
                            **r,
                            "embedding": _to_vector_literal(r["embedding"]),
                        }
                        for r in batch
                    ],
                )

    def populate_table_descriptions(
        self,
        records: List[Dict[str, Any]],  # keys: db_name, table_name, description, embedding(list[float])
        batch_size: int = 100,
    ) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur = cast(Any, cur)
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # Validate embedding dimensions
                for idx, r in enumerate(batch):
                    if len(r["embedding"]) != self.config.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch at record {i + idx}: "
                            f"expected {self.config.embedding_dim}, got {len(r['embedding'])}"
                        )
                
                cur.executemany(
                    f"""
                    INSERT INTO {self.config.schema}.{self.config.table_desc_table}
                        (db_name, table_name, description, embedding)
                    VALUES (%(db_name)s, %(table_name)s, %(description)s, %(embedding)s::vector)
                    """,
                    [
                        {
                            **r,
                            "embedding": _to_vector_literal(r["embedding"]),
                        }
                        for r in batch
                    ],
                )
    
    # ---------- Methods not implemented (not used by NLQS core) ----------
    def store_column_info_in_db(
        self,
        column_name: str,
        description: str,
        column_type: ColumnType,
    ) -> None:
        """Store column information in the database.
        
        Note: This method is not implemented as it's not used by NLQS core functionality.
        Use populate_column_info() for batch operations instead.
        """
        raise NotImplementedError(
            "store_column_info_in_db is not implemented. Use populate_column_info() for batch operations."
        )

    def qualitative_table_name_search(self, data: Dict[str, str]) -> List[str]:
        """Performs similarity search for table names.
        
        Note: This method is not implemented as it's not used by NLQS core functionality.
        """
        raise NotImplementedError(
            "qualitative_table_name_search is not implemented. "
            "This feature is not currently used by NLQS."
        )

    def qualitative_db_name_search(self, data: Dict[str, str]) -> List[str]:
        """Performs similarity search for database names.
        
        Note: This method is not implemented as it's not used by NLQS core functionality.
        """
        raise NotImplementedError(
            "qualitative_db_name_search is not implemented. "
            "This feature is not currently used by NLQS."
        )