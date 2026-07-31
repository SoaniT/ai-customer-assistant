import httpx
from .domain import RawPage

async def fetch_page(url: str, client: httpx.AsyncClient, timeout:float = 15.0) -> RawPage:
    """
    Fetches a web page and returns a RawPage object containing the URL, status code, HTML content, and content type.

    Args:
        url (str): The URL of the web page to fetch.
        client (httpx.AsyncClient): An instance of httpx.AsyncClient for making HTTP requests.
        timeout (float): The timeout for the request in seconds. Default is 15.0 seconds.

    Returns:
        RawPage: An object containing the URL, status code, HTML content, and content type.
    """
    response = await client.get(url, timeout=timeout, follow_redirects=True)
    return RawPage(
        url = str(response.url),
        status_code = response.status_code,
        html = response.text,
        content_type = response.headers.get("content-type", ""),
    )