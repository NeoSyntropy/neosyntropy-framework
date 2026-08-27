# Knowledge cookbook

Standalone examples for retrieval and transformation flows.

## Examples

- `retrieval_example.py` - search a small local corpus with `FileSystemKnowledge`
- `transform_example.py` - transform retrieved documents into a new structured summary

## Run

```bash
python cookbook/knowledge/retrieval_example.py
python cookbook/knowledge/transform_example.py
```

## Notes

- Both examples create their own temporary fixture corpus at runtime.
- No external API keys are required.
