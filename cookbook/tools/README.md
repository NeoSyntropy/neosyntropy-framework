# Tools cookbook

Standalone examples for real toolkit usage.

## Examples

- `filesystem_example.py` - write and read files with `LocalFileSystemTools`
- `web_search_example.py` - search the web with `DuckDuckGoTools` and scrape a result with `TrafilaturaTools`

## Run

```bash
python cookbook/tools/filesystem_example.py
python cookbook/tools/web_search_example.py
```

## Notes

- Both examples are self-contained and do not need API keys.
- The web example uses `ddgs` and `trafilatura`; install `ddgs` if your environment does not already have it.
