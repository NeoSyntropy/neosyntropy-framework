import asyncio
import os
import json
from pathlib import Path

# Try to load .env from the tests directory so we have the API key
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[2] / "tests" / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from neosyntropy.backend import BackendClient, BackendProvider
from neosyntropy.benchmark.synthesizer import FSMSynthesizer
from fsm import fsm

async def main():
    client = BackendClient.from_env()
    if not client:
        print("Error: NEOSYNTROPY_API_KEY environment variable is not set. Please set it in tests/.env")
        return
        
    # Using a fast model for generation
    provider = BackendProvider(client, inference_model="gemini-2.5-flash")
    
    synthesizer = FSMSynthesizer(fsm=fsm, provider=provider)
    
    # Tier 1 FSM has 6 routes. 17 samples per edge gives us 102 total cases.
    samples_per_edge = 17
    print(f"Generating {samples_per_edge} samples per route (~100 total)... This will take a few moments.")
    
    dataset = await synthesizer.synthesize_entry_cases(samples_per_edge=samples_per_edge)
    
    output_path = Path(__file__).parent / "synthesized_dataset.json"
    
    # Serialize to JSON
    out_data = []
    for case in dataset.router_cases:
        out_data.append(case.model_dump())
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"\nSuccessfully generated {len(dataset.router_cases)} samples.")
    print(f"Saved dataset to {output_path.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
