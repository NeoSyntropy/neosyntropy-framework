import os
import re

mapping = {
    "memory": ["neo4j", "zep"],
    "coding": ["coding_tools", "ast_tools", "python", "shell", "docker"],
    "version_control": ["github", "gitlab", "bitbucket"],
    "email": ["gmail", "email", "aws_ses", "resend"],
    "communication": ["whatsapp", "telegram", "discord", "webex", "zoom", "twilio", "plivo", "calling"],
    "audio_video": ["mlx_transcribe", "twelvelabs", "moviepy_video", "youtube", "spotify"],
    "search": ["duckduckgo", "baidusearch", "bravesearch", "exa", "serpapi", "serper", "tavily", "searxng", "searchapi", "websearch", "youcom", "jina", "perplexity"],
    "web_scraping": ["agentql", "apify", "browserbase", "crawl4ai", "firecrawl", "newspaper", "newspaper4k", "scrapegraph", "spider", "trafilatura", "webbrowser", "webtools"],
    "database": ["duckdb", "postgres", "redshift", "sql"],
    "project_management": ["jira", "linear", "trello", "clickup", "redmine", "todoist"],
    "finance_crypto": ["financial_datasets", "openbb", "yfinance", "evm", "valyu"],
    "social": ["hackernews", "reddit", "x"],
    "workspace": ["notion", "confluence", "calcom"],
    "data_processing": ["csv_toolkit", "pandas", "visualization", "docling"],
    "research": ["arxiv", "pubmed"],
    "crm_ecommerce": ["salesforce", "zendesk", "shopify"],
    "google": ["google_bigquery", "google_drive", "google_maps", "googlecalendar", "googlesheets"],
    "misc": ["adanos", "linkup", "scavio", "seltz", "sofya", "brandfetch", "aws_lambda", "calculator", "openrouteservice", "openweather"],
    "core": ["_local_file_utils", "api", "function", "local_file_system", "mcp_toolbox", "parallel", "registry", "sleep", "tool_registry", "toolkit", "user_control_flow", "user_feedback"]
}

# invert mapping
tool_to_cat = {}
for cat, tools in mapping.items():
    for tool in tools:
        tool_to_cat[tool] = cat

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        
        # Replace `from neosyntropy.tools.X import` with `from neosyntropy.tools.CAT.X import`
        # Also `import neosyntropy.tools.X`
        
        for tool, cat in tool_to_cat.items():
            pattern1 = r"from\s+neosyntropy\.tools\." + tool + r"\b"
            repl1 = "from neosyntropy.tools." + cat + "." + tool
            new_content = re.sub(pattern1, repl1, new_content)
            
            pattern2 = r"import\s+neosyntropy\.tools\." + tool + r"\b"
            repl2 = "import neosyntropy.tools." + cat + "." + tool
            new_content = re.sub(pattern2, repl2, new_content)

            # Special case for relative imports within tools directory:
            # from .X import -> from .CAT.X import
            # Wait, relative imports are only inside `tools/__init__.py` and tools files.
            # I should be careful to only replace `from .X` if it matches a tool name EXACTLY.
            # But the file might be anywhere. Let's just do `neosyntropy.tools.X` for now.
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
    except Exception as e:
        pass

framework_dir = r"c:\Users\avrah\PycharmProjects\NeoSyntropy.com\neosyntropy-framework"
for root, dirs, files in os.walk(framework_dir):
    # ignore .git, venv, etc
    if any(ignore in root for ignore in ['.git', 'venv', '__pycache__', 'env']):
        continue
    for file in files:
        if file.endswith('.py'):
            process_file(os.path.join(root, file))

print("Done replacing absolute imports")
