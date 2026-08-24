import pandas as pd
from typing import Type, Any, Dict, Optional
from pydantic import BaseModel, ValidationError

class MongoReader:
    """Reads from a MongoDB collection into a Pandas DataFrame and validates with Pydantic."""
    def __init__(self, connection_string: str, database: str, collection: str):
        self.connection_string = connection_string
        self.database = database
        self.collection = collection

    def read_query(self, query: Dict[str, Any], schema: Type[BaseModel]) -> pd.DataFrame:
        """
        Executes a MongoDB query and returns a pandas DataFrame where every row
        is validated against the provided Pydantic schema using strict mode.
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("Please install pymongo to use MongoReader (e.g. `pip install pymongo`)")

        client = MongoClient(self.connection_string)
        db = client[self.database]
        col = db[self.collection]
        
        cursor = col.find(query)
        
        valid_records = []
        for record in cursor:
            # Handle MongoDB _id which is an ObjectId by converting to string
            if "_id" in record:
                record["_id"] = str(record["_id"])
                
            # Strict validation ensures types exactly match the schema
            validated_instance = schema.model_validate(record, strict=True)
            valid_records.append(validated_instance.model_dump())
            
        return pd.DataFrame(valid_records)
