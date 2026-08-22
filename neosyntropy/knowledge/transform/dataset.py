from typing import Any, List, Optional
import pandas as pd
from neosyntropy.core.graph import FSM

class Input:
    """Represents a readable data source."""
    def __init__(self, name: str, data: Optional[Any] = None):
        self.name = name
        self.data = data
        
    def dataframe(self) -> pd.DataFrame:
        """Returns the data as a pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, list):
            return pd.DataFrame(self.data)
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to DataFrame")
            
    def apply_fsm(self, fsm: FSM, batch_size: int = 50) -> List[Any]:
        """
        Applies an FSM concurrently over the data in batches.
        Assumes data is iterable (e.g. list of dicts from a DataFrame).
        """
        df = self.dataframe()
        records = df.to_dict(orient="records")
        return fsm.run_batch(records, batch_size=batch_size)

class Output:
    """Represents a writable data sink."""
    def __init__(self, name: str):
        self.name = name
        self.data = None
        
    def write(self, data: Any):
        """Writes data to the sink."""
        self.data = data
        
    def write_dataframe(self, df: pd.DataFrame):
        """Writes a pandas DataFrame to the sink."""
        self.data = df
