from markdownify import markdownify
from .domain import ExtractedContent

def to_markdown(content: ExtractedContent) -> str:
    body = markdownify(content.html, heading_style="ATX").strip()
    return f"# {content.title}\n\nSource: {content.url}\n\n{body}"