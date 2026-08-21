import html as html_lib
import re
from urllib.parse import urlsplit

import trafilatura
from lxml import html as lxml_html
from trafilatura.metadata import extract_metadata
from .models import PageMeta
from .exception import ExtractionError

_TITLE_SEPARATORS = re.compile(r"\s*(?:–|—|-|\||»|:)\s*")

_TEAM_NAME_CLASS = "member-name"
_TEAM_ROLE_CLASSES = ("member-designation", "member_role", "member-role", "team-role")


def _has_class(el, token: str) -> bool:
    return token in (el.get("class") or "").split()


def _has_role_class(el) -> bool:
    return any(_has_class(el, token) for token in _TEAM_ROLE_CLASSES)


def _raw_title(html_text: str) -> str:
    """The literal <title> text (entity-unescaped). Trafilatura's metadata
    title truncates at separators ("About - Acme Co" -> "About"), which would
    break site-name derivation, so read it directly."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m is None:
        return ""
    return html_lib.unescape(re.sub(r"\s+", " ", m.group(1))).strip()


def _derive_site_name(*, title: str, html_text: str, url: str) -> str:
    """Best-effort site/company name from og:site_name, then the <title>
    (last segment of "Page - Site"), then the domain."""
    m = re.search(
        r'property=["\']og:site_name["\']\s+content=["\']([^"\']+)["\']',
        html_text,
        re.IGNORECASE,
    )
    if m and m.group(1).strip():
        return m.group(1).strip()

    if title:
        parts = _TITLE_SEPARATORS.split(title)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate:
                return candidate
        if len(title.strip()) <= 40:
            return title.strip()

    host = urlsplit(url).netloc.split(":")[0].lower()
    labels = host.split(".")
    label = labels[-2] if len(labels) > 1 else labels[0]
    return label.capitalize()


def extract_team_members(html_text: str) -> tuple[tuple[str, str], ...]:
    """Find (name, role) pairs from team-member card markup.

    Looks for elements whose class contains ``member-name`` and pairs each
    with the nearest sibling/ancestor ``member-designation`` element (common
    WordPress team-card pattern). Never raises: malformed HTML yields an
    empty tuple so extraction stays total.
    """
    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return ()

    members: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name_el in root.iter():
        if not _has_class(name_el, _TEAM_NAME_CLASS):
            continue
        name = " ".join(name_el.text_content().split()).strip()
        if not name or name.lower() in seen:
            continue

        role = ""
        parent = name_el.getparent()
        for ancestor in (parent, parent.getparent() if parent is not None else None):
            if ancestor is None:
                continue
            role_els = [
                el
                for el in ancestor.iter()
                if el is not name_el and _has_role_class(el)
            ]
            if role_els:
                role = " ".join(role_els[0].text_content().split()).strip()
                break

        members.append((name, role))
        seen.add(name.lower())

    return tuple(members)


def _enrich_markdown(html_text: str, markdown: str, url: str) -> str:
    """Add what article extraction (trafilatura) deliberately drops: the site
    name (usually in the navbar/logo) and team-member cards."""
    title = _raw_title(html_text)
    site_name = _derive_site_name(title=title, html_text=html_text, url=url)

    enriched = markdown.rstrip() + "\n"
    if site_name and site_name.lower() not in markdown.lower():
        enriched = f"# {site_name}\n\n" + enriched

    members = extract_team_members(html_text)
    if members:
        enriched += "\n## Team Members\n\n"
        for name, role in members:
            enriched += f"- {name}" + (f" — {role}" if role else "") + "\n"

    return enriched


def extract_markdown(html_text: str, url: str) -> str:
    markdown = trafilatura.extract(
        html_text,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_links=True,
        include_images=False,
        favor_precision=True,
    )
    if markdown is None:
        raise ExtractionError(f"Trafilatura could not extract content from {url}")
    return _enrich_markdown(html_text, markdown, url)

def extract_meta(html_text: str, url: str) -> PageMeta:
    meta = extract_metadata(html_text, default_url=url)
    return PageMeta(
        title=(_raw_title(html_text) or (meta.title if meta and meta.title else url)),
        canonical_url=(meta.url if meta and meta.url else url),
        description=(meta.description if meta and meta.description else ""),
    )
