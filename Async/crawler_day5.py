import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Awaitable, Tuple

import aiohttp


# =========================
# logging
# =========================
logger = logging.getLogger("crawler_day5")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# =========================
# errors
# =========================
class CrawlerError(Exception):
    pass


class TransientError(CrawlerError):
    """Timeouts, 429, 503, 5xx"""


class PermanentError(CrawlerError):
    """404, 403, 401"""


class NetworkError(CrawlerError):
    """DNS, connection refused/reset"""


class ParseError(CrawlerError):
    pass


@dataclass
class HttpStatusError(CrawlerError):
    status: int
    url: str


# =========================
# models & stats
# =========================
@dataclass
class FetchResult:
    ok: bool
    url: str
    status: Optional[int] = None
    text: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    elapsed_s: float = 0.0


@dataclass
class RetryStats:
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    successful_retries: int = 0
    retry_waits_s: List[float] = field(default_factory=list)
    permanent_error_urls: List[str] = field(default_factory=list)

    def inc_error(self, err: Exception):
        name = type(err).__name__
        self.errors_by_type[name] = self.errors_by_type.get(name, 0) + 1

    @property
    def avg_retry_wait(self) -> float:
        if not self.retry_waits_s:
            return 0.0
        return sum(self.retry_waits_s) / len(self.retry_waits_s)


# =========================
# RetryStrategy
# =========================
Coro = Callable[..., Awaitable]


class RetryStrategy:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_on: Optional[List[type]] = None,
        base_delay: float = 0.5,
    ):
        # max_retries = количество ПОВТОРОВ (не включая первый запрос)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on or [TransientError, NetworkError]
        self.base_delay = base_delay
        self.stats = RetryStats()

    def _should_retry(self, err: Exception) -> bool:
        return any(isinstance(err, t) for t in self.retry_on)

    def _delay(self, retry_number: int, err: Exception) -> float:
        # retry_number: 1..max_retries
        delay = self.base_delay * (self.backoff_factor ** (retry_number - 1))

        # 429 — увеличиваем задержку сильнее
        cause = getattr(err, "__cause__", None)
        if isinstance(err, TransientError) and isinstance(cause, HttpStatusError) and cause.status == 429:
            delay *= 2.5

        jitter = delay * 0.1
        return max(0.0, delay + random.uniform(-jitter, jitter))

    async def execute_with_retry(
        self,
        coro: Coro,
        *args,
        url: Optional[str] = None,
        **kwargs,
    ) -> Tuple[any, int]:
        """
        Возвращает (result, attempts)
        attempts = общее число попыток, включая первую
        """

        attempts = 0
        retries_done = 0  # сколько повторов уже сделали

        while True:
            attempts += 1
            try:
                return await coro(*args, **kwargs), attempts

            except Exception as e:
                self.stats.inc_error(e)

                # если НЕ ретраебельная ошибка или уже исчерпали повторы — падаем
                if (not self._should_retry(e)) or (retries_done >= self.max_retries):
                    # приклеиваем attempts, чтобы AsyncCrawler мог корректно заполнить FetchResult
                    setattr(e, "_attempts", attempts)
                    if isinstance(e, PermanentError) and url:
                        self.stats.permanent_error_urls.append(url)
                    raise

                retries_done += 1
                delay = self._delay(retries_done, e)
                self.stats.retry_waits_s.append(delay)

                logger.warning(
                    "retry | url=%s | error=%s | attempt=%d | next_in=%.2fs",
                    url, type(e).__name__, attempts, delay
                )

                await asyncio.sleep(delay)


# =========================
# AsyncCrawler
# =========================

class AsyncCrawler:
    def __init__(
        self,
        retry_strategy: RetryStrategy,
        timeout: aiohttp.ClientTimeout | None = None,
    ):
        self.retry_strategy = retry_strategy

        # БАЗОВЫЕ таймауты (на 1-ю попытку)
        self.base_timeout = timeout or aiohttp.ClientTimeout(
            total=10,
            connect=3,
            sock_connect=3,
            sock_read=5,
        )

    def _grow_timeout(self, attempt: int) -> aiohttp.ClientTimeout:
        """
        Увеличиваем таймауты при ретраях.
        attempt = номер попытки (1, 2, 3, ...)
        """
        factor = 1.5 ** (attempt - 1)
        t = self.base_timeout
        return aiohttp.ClientTimeout(
            total=(t.total * factor) if t.total else None,
            connect=(t.connect * factor) if t.connect else None,
            sock_connect=(t.sock_connect * factor) if t.sock_connect else None,
            sock_read=(t.sock_read * factor) if t.sock_read else None,
        )

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int]:
        try:
            async with session.get(url) as resp:
                status = resp.status

                if status == 404:
                    raise PermanentError() from HttpStatusError(404, url)
                if status in (401, 403):
                    raise PermanentError() from HttpStatusError(status, url)
                if status in (429, 503):
                    raise TransientError() from HttpStatusError(status, url)
                if status >= 500:
                    raise TransientError() from HttpStatusError(status, url)

                text = await resp.text()
                return text, status

        except asyncio.TimeoutError as e:
            raise TransientError() from e
        except aiohttp.ClientError as e:
            raise NetworkError() from e

    async def fetch(self, url: str) -> FetchResult:
        start = time.monotonic()

        attempt = 1
        while True:
            timeout = self._grow_timeout(attempt)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    (text, status), attempts = await self.retry_strategy.execute_with_retry(
                        self.fetch_url,
                        session,
                        url,
                        url=url,
                    )

                    elapsed = time.monotonic() - start
                    if attempts > 1:
                        self.retry_strategy.stats.successful_retries += 1

                    return FetchResult(
                        ok=True,
                        url=url,
                        status=status,
                        text=text,
                        attempts=attempts,
                        elapsed_s=elapsed,
                    )

                except Exception as e:
                    # если RetryStrategy исчерпал повторы или ошибка permanent — вернём итоговый fail
                    elapsed = time.monotonic() - start

                    status = None
                    cause = getattr(e, "__cause__", None)
                    if isinstance(cause, HttpStatusError):
                        status = cause.status

                    attempts = getattr(e, "_attempts", 1)

                    return FetchResult(
                        ok=False,
                        url=url,
                        status=status,
                        error_type=type(e).__name__,
                        error=str(e),
                        attempts=attempts,
                        elapsed_s=elapsed,
                    )
                finally:
                    attempt += 1


# =========================
# demo + report
# =========================
async def demo():
    retry = RetryStrategy(max_retries=3, backoff_factor=2.0, base_delay=0.5)
    crawler = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=5))

    urls = [
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/403",
        "https://httpbin.org/status/503",
        "https://httpbin.org/status/429",
    ]

    results: List[FetchResult] = []
    for u in urls:
        r = await crawler.fetch(u)
        results.append(r)

        logger.info(
            "done | ok=%s | url=%s | status=%s | attempts=%d | elapsed=%.2fs | err=%s",
            r.ok, r.url, r.status, r.attempts, r.elapsed_s, r.error_type
        )

    st = retry.stats
    logger.info("---- stats ----")
    logger.info("errors_by_type=%s", st.errors_by_type)
    logger.info("successful_retries=%d", st.successful_retries)
    logger.info("avg_retry_wait=%.2fs", st.avg_retry_wait)
    logger.info("permanent_error_urls=%s", st.permanent_error_urls)

    report = {
        "results": [
            {
                "ok": r.ok,
                "url": r.url,
                "status": r.status,
                "attempts": r.attempts,
                "elapsed_s": r.elapsed_s,
                "error_type": r.error_type,
                "error": r.error,
            }
            for r in results
        ],
        "stats": {
            "errors_by_type": st.errors_by_type,
            "successful_retries": st.successful_retries,
            "avg_retry_wait": st.avg_retry_wait,
            "permanent_error_urls": st.permanent_error_urls,
        },
    }

    with open("day5_error_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("report saved: day5_error_report.json")


# ---------------- main ----------------
if __name__ == "__main__":
    import sys
    import subprocess

    mode = (sys.argv[1] if len(sys.argv) > 1 else "demo").lower()

    if mode == "test":
        raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "-v", "test_crawler_day5"]))
    else:
        asyncio.run(demo())
