from __future__ import annotations

import json
from textwrap import shorten

from neosyntropy.tools.search.duckduckgo import DuckDuckGoTools
from neosyntropy.tools.web_scraping.trafilatura import TrafilaturaTools


def main() -> None:
    search_tools = DuckDuckGoTools(enable_news=False, fixed_max_results=3, timeout=15)
    scrape_tools = TrafilaturaTools(
        enable_extract_text=True,
        enable_extract_metadata_only=True,
        enable_html_to_text=False,
        enable_extract_batch=False,
        enable_crawl_website=False,
        with_metadata=True,
    )

    query = "Python pathlib Path documentation"
    raw_results = search_tools.web_search(query=query, max_results=3)
    results = json.loads(raw_results)

    print(f"Search query: {query}")
    print(f"Returned {len(results)} results")

    if not results:
        return

    first = results[0]
    url = first.get("href") or first.get("url")
    title = first.get("title") or "(untitled)"

    print()
    print(f"Top result: {title}")
    print(f"URL: {url}")

    if not url:
        print("No scrapeable URL was returned by the search engine.")
        return

    metadata = scrape_tools.extract_metadata_only(url=url, as_json=True)
    text = scrape_tools.extract_text(url=url)

    print()
    print("Metadata:")
    print(metadata)
    print()
    print("Text snippet:")
    print(shorten(text, width=800, placeholder=" ..."))


if __name__ == "__main__":
    main()
