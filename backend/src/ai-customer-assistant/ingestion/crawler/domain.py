from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RawPage:
    url: str
    status_code: int
    html: str
    content_type: Optional[str] = None

@dataclass(frozen=True)
class ExtractedContent:
    url: str
    title: Optional[str]
    html: str
    text: str

@dataclass(frozen=True)
class CrawlResult:
    url: str
    title: Optional[str]
    markdown: str
    html: str
    links: tuple[str, ...] = ()
    error: Optional[str] = None
