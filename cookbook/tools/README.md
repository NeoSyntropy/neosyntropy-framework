# Tools cookbook

Standalone examples for real toolkit usage.

## Examples

- `filesystem_example.py` - write and read files with `LocalFileSystemTools`
- `web_search_example.py` - search the web with `DuckDuckGoTools` and scrape a result with `TrafilaturaTools`
- `email_example.py` - send an email with `EmailTools`

## Run

```bash
python cookbook/tools/filesystem_example.py
python cookbook/tools/web_search_example.py
python cookbook/tools/email_example.py
```

## Notes

- The filesystem and web examples are self-contained and do not need API keys.
- The web example uses `ddgs` and `trafilatura`; install `ddgs` if your environment does not already have it.
- The email example sends a real message through Gmail SMTP.
- Set `RECIPIENT_EMAIL`, `SENDER_NAME`, `SENDER_EMAIL`, and `SENDER_PASSKEY` before running `email_example.py`.
- Use a Gmail app password for `SENDER_PASSKEY`.
