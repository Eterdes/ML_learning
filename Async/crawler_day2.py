import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import aiohttp
from html_parser import HTMLParser
from bs4 import BeautifulSoup


# ---------- logging ----------
logger = logging.getLogger("crawler_day1")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------- crawler ----------
@dataclass
class FetchResult:
    ok: bool
    status: Optional[int]
    text: Optional[str]
    error: Optional[str]


class AsyncCrawler:
    """
    Базовый асинхронный HTTP-клиент:
    - ограничение конкурентности через Semaphore
    - ClientSession с таймаутами + connection pooling
    - обработка ClientError / TimeoutError / ClientResponseError
    """

    def __init__(self, max_concurrent: int = 10, connect_timeout: float = 5.0, read_timeout: float = 10.0):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._parser = HTMLParser()

        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=read_timeout,
        )

        # connection pooling (TCPConnector):
        # limit - общий лимит одновременных соединений
        # limit_per_host - лимит на хост (удобно, чтобы не DDOSить один домен)
        connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max(1, max_concurrent // 2))

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            raise_for_status=True,  # будет бросать ClientResponseError на 4xx/5xx
            headers={"User-Agent": "crawler_day1/1.0"},
        )

    async def fetch_url(self, url: str) -> tuple[int, str]:
        async with self._sem:
            logger.info(f"▶️  START {url}")
            try:
                async with self._session.get(url) as resp:
                   text = await resp.text(errors="replace")
                   logger.info(
                       f"✅ DONE  {url} (status={resp.status}, bytes={len(text)})"
                       )
                   return resp.status, text
            except aiohttp.ClientResponseError as e:
                logger.warning(f"⚠️  HTTP_ERROR {url} (status={e.status}) {type(e).__name__}: {e}")
                raise
            except asyncio.TimeoutError as e:
                logger.warning(f"⚠️  TIMEOUT {url} {type(e).__name__}: {e}")
                raise
            except aiohttp.ClientError as e:
                logger.warning(f"⚠️  CLIENT_ERROR {url} {type(e).__name__}: {e}")
                raise

    async def fetch_urls(self, urls: List[str]) -> Dict[str, FetchResult]:
        """
        Параллельная загрузка URL.
        Возвращает dict[url] = FetchResult, чтобы ошибки не роняли программу.
        """
        async def _safe_fetch(u: str) -> FetchResult:
            try:
                status, text = await self.fetch_url(u)
                return FetchResult(ok=True, status=status, text=text, error=None)
            except aiohttp.ClientResponseError as e:
                return FetchResult(ok=False, status=e.status, text=None, error=f"{type(e).__name__}: {e}")
            except asyncio.TimeoutError as e:
                return FetchResult(ok=False, status=None, text=None, error=f"{type(e).__name__}: {e}")
            except aiohttp.ClientError as e:
                return FetchResult(ok=False, status=None, text=None, error=f"{type(e).__name__}: {e}")
            except Exception as e:
                # на всякий случай
                return FetchResult(ok=False, status=None, text=None, error=f"Unexpected {type(e).__name__}: {e}")

        tasks = [asyncio.create_task(_safe_fetch(u)) for u in urls]
        results_list = await asyncio.gather(*tasks)
        return {u: r for u, r in zip(urls, results_list)}

    async def close(self) -> None:
        await self._session.close()

    async def fetch_and_parse(self, url: str) -> dict:
        fetched = await self.fetch(url)

        base = {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "metadata": {},
            "images": [],
            "headings": {"h1": [], "h2": [], "h3": []},
            "tables": [],
            "lists": {"ul": [], "ol": []},
            "warnings": [],
            "fetch": {
                "ok": fetched.ok,
                "status": fetched.status,
                "error": fetched.error,
            },
        }

        if not fetched.ok or not fetched.text:
            return base

        parsed = await self._parser.parse_html(fetched.text, url)

        base["title"] = parsed.get("title", "")
        base["text"] = parsed.get("text", "")
        base["links"] = parsed.get("links", [])
        base["metadata"] = parsed.get("metadata", {})
        base["images"] = parsed.get("images", [])
        base["headings"] = parsed.get("headings", {})
        base["tables"] = parsed.get("tables", [])
        base["lists"] = parsed.get("lists", {})
        base["warnings"] = parsed.get("warnings", [])

        return base

# ---------- demo / tests ----------
async def run_day2_tests():
    p = HTMLParser()

    # 1) валидный HTML
    valid_html = """
    <html>
      <head>
        <title>Example</title>
        <meta name="description" content="Desc here">
      </head>
      <body>
        <h1>Main</h1>
        <p>Hello <b>world</b></p>
        <a href="/page1">rel</a>
        <a href="https://other.com/x">abs</a>
        <img src="/img.png" alt="pic">
      </body>
    </html>
    """
    res1 = await p.parse_html(valid_html, "https://site.com/base")
    assert res1["title"] == "Example"
    assert "Hello" in res1["text"]
    assert "https://site.com/page1" in res1["links"]
    assert "https://other.com/x" in res1["links"]
    assert res1["metadata"]["description"] == "Desc here"
    assert res1["images"][0]["src"] == "https://site.com/img.png"
    assert res1["images"][0]["alt"] == "pic"

    # 2) битый HTML (проверка: не падает)
    broken_html = "<html><head><title>Broken<title></head><body><h1>H1<p>Oops<a href='/a'>"
    res2 = await p.parse_html(broken_html, "https://site.com/")
    assert isinstance(res2, dict)
    assert isinstance(res2["links"], list)
    assert res2["url"] == "https://site.com/"

    # 3) извлечение ссылок + относительные URL
    soup = BeautifulSoup('<a href="/x">x</a><a href="mailto:a@b.com">m</a>', "lxml")
    links = p.extract_links(soup, "https://site.com/base")
    assert "https://site.com/x" in links
    assert all(not l.startswith("mailto:") for l in links)

    print("✅ Day 2 parser tests passed")

if __name__ == "__main__":
    asyncio.run(run_day2_tests())