import os
from pathlib import Path

tests_env = Path("tests/.env")
for line in tests_env.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    os.environ[k.strip()] = v.strip().strip("'\"")

from neosyntropy import Client

c = Client(api_key=os.environ["NEOSYNTROPY_API_KEY"], base_url="http://127.0.0.1:8000")
projects = c.list_projects()
print(f"Total projects: {len(projects)}")
for p in projects[:10]:
    print(f"  id={p.get('id')}  slug={p.get('slug')!r}  name={p.get('name')!r}")
