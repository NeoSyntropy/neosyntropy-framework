"""Intentional secret leak for the CI secret-scan demo (FAKE values only)."""

from neosyntropy import Client

# BAD: hardcoded NeoSyntropy key (mirrors the anti-pattern CI should catch).
client = Client(
    api_key="nsk_EXAMPLE_LEAKED_KEY_DO_NOT_USE_abc123xyz789",
    project_id="00000000-0000-0000-0000-000000000000",
    base_url="https://api.neosyntropy.com",
)
