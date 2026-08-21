import pytest

from ingestion.crawler.extractor import (
    _derive_site_name,
    extract_markdown,
    extract_team_members,
)

_ABOUT_HTML = """<!DOCTYPE html>
<html>
<head><title>About - Acme Co</title></head>
<body>
  <nav><a href="/"><img alt="Acme Co logo"></a></nav>
  <h1>Meet the Team</h1>
  <div class="team-member-wrapper">
    <div class="member-name">Jane Doe</div>
    <div class="member-designation pb-3">CEO, Acme Co</div>
  </div>
  <div class="team-member-wrapper">
    <div class="member-name">John Smith</div>
    <div class="member-designation">CTO</div>
  </div>
  <p>Our vision is to lead.</p>
</body>
</html>
"""

_PLAIN_HTML = "<html><head><title>Just a Page</title></head><body><p>Some body text.</p></body></html>"


def test_derive_site_name_from_title_last_segment():
    assert _derive_site_name(title="About - Acme Co", html_text="", url="https://acme.co/about/") == "Acme Co"


def test_derive_site_name_from_og_site_name():
    html = '<meta property="og:site_name" content="Acme Co">'
    assert _derive_site_name(title="Whatever", html_text=html, url="https://acme.co/") == "Acme Co"


def test_derive_site_name_from_domain_fallback():
    assert _derive_site_name(title="", html_text="", url="https://acme.co/about/") == "Acme"


def test_extract_team_members_parses_cards():
    assert extract_team_members(_ABOUT_HTML) == (
        ("Jane Doe", "CEO, Acme Co"),
        ("John Smith", "CTO"),
    )


def test_extract_team_members_empty_and_malformed():
    assert extract_team_members("<html><body><p>no cards</p></body></html>") == ()
    assert extract_team_members("not html at all <<<") == ()


def test_extract_markdown_enriches_with_site_name_and_members():
    md = extract_markdown(_ABOUT_HTML, "https://acme.co/about/")
    assert md.startswith("# Acme Co")
    assert "Jane Doe" in md and "CEO, Acme Co" in md
    assert "John Smith" in md and "CTO" in md
    assert "Some body text." not in md  # sanity: unrelated fixture stays unenriched below


def test_extract_markdown_no_members_plain_page():
    md = extract_markdown(_PLAIN_HTML, "https://acme.co/")
    assert md.startswith("# Just a Page")
    assert "Team Members" not in md


def test_extract_markdown_does_not_duplicate_site_name():
    html = '<html><head><title>Acme Co</title></head><body><h1>Acme Co</h1><p>About us.</p></body></html>'
    md = extract_markdown(html, "https://acme.co/")
    assert md.count("# Acme Co") == 1
