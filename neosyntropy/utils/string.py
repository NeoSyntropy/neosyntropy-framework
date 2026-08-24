import uuid

def generate_uuid() -> str:
    return str(uuid.uuid4())

def generate_id(length: int = 8) -> str:
    return str(uuid.uuid4())[:length]
