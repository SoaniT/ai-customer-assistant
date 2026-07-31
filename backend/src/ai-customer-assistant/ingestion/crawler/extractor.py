from urllib.parse import urljoin
from bs4 import BeautifulSoup
from .domain import RawPage, ExtractedContent

STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg")

def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    tuple(tag.decompose() for tag in soup.find_all(STRIP_TAGS))
    return soup

def extract_links(soup: BeautifulSoup, base_url: str) -> tuple[str, ...]:
    return tuple(sorted({
        urljoin(base_url, a["href"])
        for a in soup.find_all("a", href=True)
        if not a["href"].startswith(("mailto:", "javascript:", "#"))
    }))

def extract_content(page: RawPage) -> ExtractedContent:
    soup = clean_soup(page.html)
    title = (soup.title.string.strip() if soup.title and soup.title.string else page.url)
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return ExtractedContent(
        url=page.url, title=title, html=str(main), text=main.get_text(" ", strip=True)
    )