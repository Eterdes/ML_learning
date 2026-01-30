import abc
import asyncio
import csv
import io
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Tuple, Any

import aiohttp
import aiofiles
import aiosqlite


# =========================
# logging
# =========================
logger = logging.getLogger("crawler_day6")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# =========================
# errors (from day5)
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
    content_type: Optional[str] = None
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
# RetryStrategy (from day5)
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
    ) -> Tuple[Any, int]:
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

                if (not self._should_retry(e)) or (retries_done >= self.max_retries):
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
# Day6: schema + normalize
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


EXPECTED_KEYS = {
    "url", "title", "text", "links", "metadata", "crawled_at", "status_code", "content_type"
}


def normalize_record(data: Dict[str, Any]) -> Dict[str, Any]:
    crawled_at = data.get("crawled_at") or utc_now()
    if isinstance(crawled_at, datetime):
        crawled_at = to_iso(crawled_at)

    links = data.get("links") or []
    if not isinstance(links, list):
        links = list(links)

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    return {
        "url": str(data.get("url") or ""),
        "title": str(data.get("title") or ""),
        "text": str(data.get("text") or ""),
        "links": [str(x) for x in links],
        "metadata": metadata,
        "crawled_at": str(crawled_at),
        "status_code": int(data.get("status_code") or 0),
        "content_type": str(data.get("content_type") or ""),
    }


def validate_schema(record: dict) -> None:
    missing = EXPECTED_KEYS - set(record.keys())
    if missing:
        raise AssertionError(f"missing keys: {missing}")

    if not isinstance(record["url"], str):
        raise AssertionError("url must be str")
    if not isinstance(record["title"], str):
        raise AssertionError("title must be str")
    if not isinstance(record["text"], str):
        raise AssertionError("text must be str")
    if not isinstance(record["links"], list):
        raise AssertionError("links must be list")
    if not isinstance(record["metadata"], dict):
        raise AssertionError("metadata must be dict")
    if not isinstance(record["crawled_at"], str):
        raise AssertionError("crawled_at must be str")
    if not isinstance(record["status_code"], int):
        raise AssertionError("status_code must be int")
    if not isinstance(record["content_type"], str):
        raise AssertionError("content_type must be str")


# =========================
# Day6: Storage
# =========================
class DataStorage(abc.ABC):
    @abc.abstractmethod
    async def save(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class JSONStorage(DataStorage):
    """
    NDJSON: одна запись = одна строка JSON (масштабируется на большие объёмы).
    pretty=True делает indent у каждого объекта, но всё равно пишем построчно.
    """

    def __init__(self, path: str, encoding: str = "utf-8", ensure_ascii: bool = False, pretty: bool = False,
                 flush_every: int = 50):
        self.path = path
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
        self.pretty = pretty
        self.flush_every = max(1, flush_every)

        self._file = None
        self._n = 0
        self._lock = asyncio.Lock()

    async def _ensure_open(self) -> None:
        if self._file is None:
            self._file = await aiofiles.open(self.path, "a", encoding=self.encoding)

    async def save(self, data: Dict[str, Any]) -> None:
        item = normalize_record(data)
        async with self._lock:
            await self._ensure_open()
            line = json.dumps(item, ensure_ascii=self.ensure_ascii, indent=2 if self.pretty else None)
            await self._file.write(line + "\n")
            self._n += 1
            if self._n % self.flush_every == 0:
                await self._file.flush()

    async def close(self) -> None:
        async with self._lock:
            if self._file is not None:
                await self._file.flush()
                await self._file.close()
                self._file = None


class CSVStorage(DataStorage):
    """
    - Заголовки определяем по первой записи
    - links/metadata сохраняем как JSON-строки (иначе CSV неудобен)
    """

    def __init__(self, path: str, encoding: str = "utf-8", delimiter: str = ",", flush_every: int = 50):
        self.path = path
        self.encoding = encoding
        self.delimiter = delimiter
        self.flush_every = max(1, flush_every)

        self._file = None
        self._headers: Optional[List[str]] = None
        self._n = 0
        self._lock = asyncio.Lock()

    async def _ensure_open(self) -> None:
        if self._file is None:
            self._file = await aiofiles.open(self.path, "a", encoding=self.encoding, newline="")

    def _to_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(item)
        row["links"] = json.dumps(row.get("links", []), ensure_ascii=False)
        row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
        return row

    async def save(self, data: Dict[str, Any]) -> None:
        item = normalize_record(data)
        row = self._to_row(item)

        async with self._lock:
            await self._ensure_open()

            if self._headers is None:
                self._headers = list(row.keys())
                buf = io.StringIO()
                w = csv.DictWriter(
                    buf,
                    fieldnames=self._headers,
                    delimiter=self.delimiter,
                    quoting=csv.QUOTE_MINIMAL,
                    lineterminator="\n",
                )
                w.writeheader()
                await self._file.write(buf.getvalue())

            buf2 = io.StringIO()
            w2 = csv.DictWriter(
                buf2,
                fieldnames=self._headers,
                delimiter=self.delimiter,
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            w2.writerow(row)
            await self._file.write(buf2.getvalue())

            self._n += 1
            if self._n % self.flush_every == 0:
                await self._file.flush()

    async def close(self) -> None:
        async with self._lock:
            if self._file is not None:
                await self._file.flush()
                await self._file.close()
                self._file = None


class SQLiteStorage(DataStorage):
    """
    SQLite + batch-вставки (executemany) + индексы
    """

    def __init__(self, path: str, table: str = "pages", batch_size: int = 100, flush_interval: float = 1.0):
        self.path = path
        self.table = table
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.1, flush_interval)

        self._db: Optional[aiosqlite.Connection] = None
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flusher: Optional[asyncio.Task] = None
        self._closed = False

    async def init_db(self) -> None:
        if self._db is not None:
            return

        self._db = await aiosqlite.connect(self.path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")

        await self._db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT,
                text TEXT,
                links_json TEXT,
                metadata_json TEXT,
                crawled_at TEXT,
                status_code INTEGER,
                content_type TEXT
            );
            """
        )
        await self._db.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_url ON {self.table}(url);")
        await self._db.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_crawled ON {self.table}(crawled_at);")
        await self._db.commit()

        self._flusher = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            return

    async def flush(self) -> None:
        async with self._lock:
            if self._db is None or not self._buffer:
                return
            batch = self._buffer
            self._buffer = []

        rows = [
            (
                x["url"],
                x["title"],
                x["text"],
                json.dumps(x["links"], ensure_ascii=False),
                json.dumps(x["metadata"], ensure_ascii=False),
                x["crawled_at"],
                x["status_code"],
                x["content_type"],
            )
            for x in batch
        ]

        await self._db.executemany(
            f"""
            INSERT INTO {self.table}
            (url, title, text, links_json, metadata_json, crawled_at, status_code, content_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        await self._db.commit()

    async def save(self, data: Dict[str, Any]) -> None:
        await self.init_db()
        item = normalize_record(data)

        async with self._lock:
            self._buffer.append(item)
            need_flush = len(self._buffer) >= self.batch_size

        if need_flush:
            await self.flush()

    async def close(self) -> None:
        self._closed = True
        if self._flusher is not None:
            self._flusher.cancel()
            try:
                await self._flusher
            except Exception:
                pass
            self._flusher = None

        await self.flush()
        if self._db is not None:
            await self._db.close()
            self._db = None


# =========================
# Day6: save retry helper (для ошибок записи)
# =========================
async def save_with_retry(
    storage: DataStorage,
    record: Dict[str, Any],
    *,
    max_retries: int = 3,
    base_delay: float = 0.2,
    backoff_factor: float = 2.0,
) -> None:
    attempt = 0
    while True:
        try:
            await storage.save(record)
            return
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = base_delay * (backoff_factor ** (attempt - 1))
            jitter = delay * 0.1
            delay = max(0.0, delay + random.uniform(-jitter, jitter))
            logger.warning("save retry | attempt=%d | next_in=%.2fs | error=%s", attempt, delay, type(e).__name__)
            await asyncio.sleep(delay)


# =========================
# Day6: read-back helpers (demo + useful)
# =========================
async def read_back_ndjson(path: str, limit: int = 3) -> List[dict]:
    items: List[dict] = []
    if not Path(path).exists():
        return items

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            validate_schema(obj)
            items.append(obj)
            if len(items) >= limit:
                break
    return items


async def read_back_csv(path: str, limit: int = 3) -> List[dict]:
    items: List[dict] = []
    if not Path(path).exists():
        return items

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()

    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        row["links"] = json.loads(row.get("links") or "[]")
        row["metadata"] = json.loads(row.get("metadata") or "{}")

        try:
            row["status_code"] = int(row.get("status_code") or 0)
        except ValueError:
            row["status_code"] = 0

        validate_schema(row)
        items.append(row)
        if len(items) >= limit:
            break

    return items


async def read_back_sqlite(path: str, table: str = "pages", limit: int = 3) -> List[dict]:
    items: List[dict] = []
    if not Path(path).exists():
        return items

    async with aiosqlite.connect(path) as db:
        q = f"""
        SELECT url, title, text, links_json, metadata_json, crawled_at, status_code, content_type
        FROM {table}
        ORDER BY id DESC
        LIMIT ?;
        """
        async with db.execute(q, (limit,)) as cur:
            rows = await cur.fetchall()

    for (url, title, text, links_json, metadata_json, crawled_at, status_code, content_type) in rows:
        obj = {
            "url": url or "",
            "title": title or "",
            "text": text or "",
            "links": json.loads(links_json or "[]"),
            "metadata": json.loads(metadata_json or "{}"),
            "crawled_at": crawled_at or "",
            "status_code": int(status_code or 0),
            "content_type": content_type or "",
        }
        validate_schema(obj)
        items.append(obj)

    return items


# =========================
# AsyncCrawler (твоя логика + storage интеграция)
# =========================
class AsyncCrawler:
    def __init__(
        self,
        retry_strategy: RetryStrategy,
        timeout: aiohttp.ClientTimeout | None = None,
        storage: Optional[DataStorage] = None,
        save_retries: int = 3,
    ):
        self.retry_strategy = retry_strategy
        self.storage = storage
        self.save_retries = save_retries

        self.base_timeout = timeout or aiohttp.ClientTimeout(
            total=10,
            connect=3,
            sock_connect=3,
            sock_read=5,
        )

        self.saved_ok = 0
        self.saved_failed = 0

    def _grow_timeout(self, attempt: int) -> aiohttp.ClientTimeout:
        factor = 1.5 ** (attempt - 1)
        t = self.base_timeout
        return aiohttp.ClientTimeout(
            total=(t.total * factor) if t.total else None,
            connect=(t.connect * factor) if t.connect else None,
            sock_connect=(t.sock_connect * factor) if t.sock_connect else None,
            sock_read=(t.sock_read * factor) if t.sock_read else None,
        )

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int, str]:
        try:
            async with session.get(url) as resp:
                status = resp.status
                ct = resp.headers.get("content-type", "")

                if status == 404:
                    raise PermanentError() from HttpStatusError(404, url)
                if status in (401, 403):
                    raise PermanentError() from HttpStatusError(status, url)
                if status in (429, 503):
                    raise TransientError() from HttpStatusError(status, url)
                if status >= 500:
                    raise TransientError() from HttpStatusError(status, url)

                text = await resp.text()
                return text, status, ct

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
                    (text, status, ct), attempts = await self.retry_strategy.execute_with_retry(
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
                        content_type=ct,
                        attempts=attempts,
                        elapsed_s=elapsed,
                    )

                except Exception as e:
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

    def _build_record_from_fetch(self, r: FetchResult) -> Dict[str, Any]:
        return {
            "url": r.url,
            "title": "",
            "text": r.text or "",
            "links": [],
            "metadata": {
                "ok": r.ok,
                "attempts": r.attempts,
                "elapsed_s": r.elapsed_s,
                "error_type": r.error_type,
                "error": r.error,
            },
            "crawled_at": utc_now(),
            "status_code": r.status or 0,
            "content_type": r.content_type or "",
        }

    async def crawl_and_save(self, urls: List[str]) -> List[FetchResult]:
        results: List[FetchResult] = []

        for u in urls:
            r = await self.fetch(u)
            results.append(r)

            if self.storage is not None:
                record = self._build_record_from_fetch(r)
                try:
                    await save_with_retry(self.storage, record, max_retries=self.save_retries)
                    self.saved_ok += 1
                except Exception:
                    self.saved_failed += 1
                    logger.exception("storage save failed (continue) | url=%s", u)

        return results

    async def close(self) -> None:
        if self.storage is not None:
            await self.storage.close()


# =========================
# demo + report (Day6)
# =========================
async def demo():
    retry = RetryStrategy(max_retries=3, backoff_factor=2.0, base_delay=0.5)

    urls = [
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/403",
        "https://httpbin.org/status/503",
        "https://httpbin.org/status/429",
    ]

    # ---- JSON ----
    logger.info("=== JSONStorage demo ===")
    json_storage = JSONStorage("results.ndjson", pretty=False, flush_every=1)
    crawler_json = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=5), storage=json_storage)
    results = await crawler_json.crawl_and_save(urls)
    await crawler_json.close()
    logger.info("saved_ok=%d saved_failed=%d", crawler_json.saved_ok, crawler_json.saved_failed)

    back = await read_back_ndjson("results.ndjson", limit=3)
    logger.info("read-back JSON: %d records | sample_urls=%s", len(back), [x["url"] for x in back])

    # ---- CSV ----
    logger.info("=== CSVStorage demo ===")
    csv_storage = CSVStorage("results.csv", flush_every=1)
    crawler_csv = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=5), storage=csv_storage)
    await crawler_csv.crawl_and_save(urls)
    await crawler_csv.close()
    logger.info("saved_ok=%d saved_failed=%d", crawler_csv.saved_ok, crawler_csv.saved_failed)

    back = await read_back_csv("results.csv", limit=3)
    logger.info("read-back CSV: %d records | sample_urls=%s", len(back), [x["url"] for x in back])

    # ---- SQLite ----
    logger.info("=== SQLiteStorage demo ===")
    db_storage = SQLiteStorage("crawler.db", batch_size=2, flush_interval=0.2)
    crawler_db = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=5), storage=db_storage)
    await crawler_db.crawl_and_save(urls)
    await crawler_db.close()
    logger.info("saved_ok=%d saved_failed=%d", crawler_db.saved_ok, crawler_db.saved_failed)

    back = await read_back_sqlite("crawler.db", table="pages", limit=3)
    logger.info("read-back SQLite: %d records | sample_urls=%s", len(back), [x["url"] for x in back])

    # report like day5 + storage stats
    st = retry.stats
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
        "retry_stats": {
            "errors_by_type": st.errors_by_type,
            "successful_retries": st.successful_retries,
            "avg_retry_wait": st.avg_retry_wait,
            "permanent_error_urls": st.permanent_error_urls,
        },
        "storage_stats": {
            "json_saved_ok": crawler_json.saved_ok,
            "json_saved_failed": crawler_json.saved_failed,
        },
        "read_back_samples": {
            "json_urls": [x["url"] for x in (await read_back_ndjson("results.ndjson", limit=3))],
            "csv_urls": [x["url"] for x in (await read_back_csv("results.csv", limit=3))],
            "sqlite_urls": [x["url"] for x in (await read_back_sqlite("crawler.db", table="pages", limit=3))],
        }
    }

    with open("day6_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("report saved: day6_report.json")


# ---------------- main ----------------
if __name__ == "__main__":
    import sys
    import subprocess

    mode = (sys.argv[1] if len(sys.argv) > 1 else "demo").lower()

    if mode == "test":
        raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "-v", "test_crawler_day6"]))
    else:
        asyncio.run(demo())