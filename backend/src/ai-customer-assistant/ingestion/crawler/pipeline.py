import asyncio
import httpx
from .domain import CrawlResult
from .fetcher import fetch_page
from .extractor import extract_content, extract_links, clean_soup
from .converter import to_markdown

async def crawl_url(url: str, client: httpx.AsyncClient) -> CrawlResult:
    """Crawls a single URL, extracting its content, links, and metadata.

    Args:
        url (str): The absolute URL to crawl.
        client (httpx.AsyncClient): An active, shared asynchronous HTTP client 
            used to perform the network request (enabling connection pooling).

    Returns:
        CrawlResult: A domain object containing the extracted title, Markdown 
        content, cleaned HTML, and a list of discovered links. If an error 
        occurred during processing, the content fields will be empty strings 
        and the `error` attribute will contain the string representation of 
        the exception.
    """
    try:
        page = await fetch_page(url, client)
        content = extract_content(page)
        links = extract_links(clean_soup(page.html), page.url)
        return CrawlResult(
            url=page.url, title=content.title,
            markdown=to_markdown(content), html=content.html, links=links,
        )
    except Exception as e:
        return CrawlResult(url=url, title="", markdown="", html="", error=str(e))

async def crawl_many(urls: tuple[str, ...], client: httpx.AsyncClient | None = None) -> tuple[CrawlResult, ...]:
    """Concurrently crawls a collection of URLs and returns their results.

    Orchestrates the concurrent fetching and processing of multiple URLs using 
    `asyncio.gather`. It intelligently manages the lifecycle of the 
    `httpx.AsyncClient`: if an existing client is passed in, it reuses it; 
    otherwise, it creates a new one and ensures it is properly closed when 
    the operation finishes.

    Args:
        urls (tuple[str, ...]): A tuple of absolute URLs to be crawled.
        client (httpx.AsyncClient | None, optional): An existing async HTTP 
            client to reuse for connection pooling. If `None`, a new client 
            will be instantiated and automatically closed upon completion. 
            Defaults to `None`.

    Returns:
        tuple[CrawlResult, ...]: A tuple of `CrawlResult` objects corresponding 
        to the input URLs. The order of the returned results strictly matches 
        the order of the input `urls` tuple.
    """
    owns_client = client is None
    active_client = client or httpx.AsyncClient()
    try:
        return tuple(await asyncio.gather(*map(lambda u: crawl_url(u, active_client), urls)))
    finally:
        if owns_client:
            await active_client.aclose()