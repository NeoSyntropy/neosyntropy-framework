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

## Remote loaders (S3, GCS, Azure, SharePoint, GitHub)

`Knowledge` inherits provider loaders. Point `Content.remote_content` at a
cloud file or folder (`S3Content`, `GitHubContent`, …). Each file gets a
content hash (MD5 of `path` / `url` / `name`) that is also used as the
storage `id`.

- `skip_if_exists=True` skips a file when a vector DB reports that hash
  already exists. The skip check takes `(content, upsert, skip_if_exists)` —
  passing the hash string alone raises `TypeError`.
- GitHub can ingest without a vector DB: documents stay on
  `Knowledge.contents` and `search()` does substring match.
- Provider metadata is stored under `_neosyntropy` so user updates cannot
  overwrite source_type / bucket / repo fields.

See [`docs/concepts-explained.md`](../../docs/concepts-explained.md#14-knowledge--corpus-and-etl).
