from pathlib import Path
from .domain import CrawlResult

FORMAT_TO_FIELD = {"md": "markdown", "html": "html"}

def _slug(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1] or "index"

def save_result(result: CrawlResult, out_dir: Path, fmt: str = "md") -> Path:
    content = getattr(result, FORMAT_TO_FIELD[fmt])
    path = out_dir / f"{_slug(result.url)}.{fmt}"
    path.write_text(content, encoding="utf-8")
    return path

def save_all(results: tuple[CrawlResult, ...], out_dir: Path, fmt: str = "md") -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return tuple(save_result(r, out_dir, fmt) for r in results if r.error is None)