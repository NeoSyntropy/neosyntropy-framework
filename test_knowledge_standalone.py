import asyncio
from neosyntropy.knowledge.filesystem import FileSystemKnowledge
from neosyntropy.databases.vector.base import VectorDb

async def main():
    print("Testing FileSystemKnowledge initialization...")
    fs_knowledge = FileSystemKnowledge(base_dir=".")
    
    print("Testing getting tools without an Agent...")
    tools = fs_knowledge.get_tools()
    print(f"Tools available: {[t.__name__ if hasattr(t, '__name__') else str(t) for t in tools]}")
    
    print("\nTesting VectorDb initialization...")
    # Mock class implementation for vector db
    class MockVectorDb(VectorDb):
        def insert(self, *args, **kwargs):
            pass
        def search(self, *args, **kwargs):
            pass
            
    vdb = MockVectorDb()
    print("All components initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
