import asyncio
import sys
from pathlib import Path
from .pipeline import crawl_many
from .io_output import save_all

async def main():
    urls = tuple(sys.argv[1:])
    results = await crawl_many(urls)
    saved = save_all(results, Path("./output"), fmt="md")
    tuple(print(f"saved: {p}") for p in saved)
    tuple(print(f"failed: {r.url} — {r.error}") for r in results if r.error)

if __name__ == "__main__":
    asyncio.run(main())