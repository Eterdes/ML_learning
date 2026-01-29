import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import aiohttp


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


# ---------- demo / tests ----------
async def sequential_fetch(crawler: AsyncCrawler, urls: List[str]) -> Dict[str, FetchResult]:
    out: Dict[str, FetchResult] = {}
    for u in urls:
        res = await crawler.fetch_urls([u])  # reuse safe wrapper
        out[u] = res[u]
    return out


def print_summary(title: str, results: Dict[str, FetchResult], elapsed: float) -> None:
    ok = sum(1 for r in results.values() if r.ok)
    fail = len(results) - ok
    print(f"\n=== {title} ===")
    print(f"Elapsed: {elapsed:.3f}s | OK: {ok} | FAIL: {fail}")
    for url, r in results.items():
        if r.ok:
            # status в этом простом варианте фиксирован 200, но можно сохранять реальный
            print(f"  ✅ {url} -> OK (bytes={len(r.text or '')})")
        else:
            print(f"  ❌ {url} -> FAIL (status={r.status}) {r.error}")


async def main():
    # 5-10 URL: валидные, 404, таймаут
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/status/404",   # HTTP ошибка
        "https://httpbin.org/status/500",   # HTTP ошибка
        "https://10.255.255.1/",            # обычно долго/не доступен (может дать timeout/client error)
    ]

    # ---- parallel ----
    crawler = AsyncCrawler(max_concurrent=5, connect_timeout=5.0, read_timeout=10.0)
    t0 = time.perf_counter()
    par_results = await crawler.fetch_urls(urls)
    t1 = time.perf_counter()
    await crawler.close()
    print_summary("PARALLEL", par_results, t1 - t0)

    # ---- sequential ----
    crawler2 = AsyncCrawler(max_concurrent=1, connect_timeout=5.0, read_timeout=10.0)
    t2 = time.perf_counter()
    seq_results = await sequential_fetch(crawler2, urls)
    t3 = time.perf_counter()
    await crawler2.close()
    print_summary("SEQUENTIAL", seq_results, t3 - t2)

    # ---- simple checks (мини-тесты без pytest) ----
    # валидные
    assert par_results["https://example.com"].ok is True
    # 404
    assert par_results["https://httpbin.org/status/404"].ok is False
    # таймаут/ошибка сети для недоступного IP (может быть client error или timeout)
    assert par_results["https://10.255.255.1/"].ok is False

    print("\n✅ Basic tests passed")


if __name__ == "__main__":
    asyncio.run(main())