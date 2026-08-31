import pandas as pd
from typing import Type, Any, Optional
from pydantic import BaseModel, ValidationError

class PostgresReader:
    """Reads from a PostgreSQL database into a Pandas DataFrame and validates with Pydantic."""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def read_sql(self, query: str, schema: Type[BaseModel]) -> pd.DataFrame:
        """
        Executes a SQL query and returns a pandas DataFrame where every row
        is validated against the provided Pydantic schema using strict mode.
        """
        try:
            import sqlalchemy
        except ImportError:
            raise ImportError("Please install sqlalchemy to use PostgresReader (e.g. `pip install sqlalchemy psycopg2-binary`)")

        engine = sqlalchemy.create_engine(self.connection_string)
        df = pd.read_sql(query, engine)
        
        valid_records = []
        for record in df.to_dict(orient="records"):
            # Strict validation ensures types exactly match the schema
            validated_instance = schema.model_validate(record, strict=True)
            valid_records.append(validated_instance.model_dump())
            
        return pd.DataFrame(valid_records)
