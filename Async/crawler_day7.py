import abc
import argparse
import asyncio
import csv
import io
import json
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiofiles
import aiohttp
import aiosqlite

# =========================
# logging
# =========================
def setup_logging(log_file: str = "crawler.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("crawler")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logging()


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
                    url,
                    type(e).__name__,
                    attempts,
                    delay,
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
    "url",
    "title",
    "text",
    "links",
    "metadata",
    "crawled_at",
    "status_code",
    "content_type",
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
    NDJSON: одна запись = одна строка JSON.
    """

    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        ensure_ascii: bool = False,
        pretty: bool = False,
        flush_every: int = 50,
    ):
        self.path = path
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
        self.pretty = pretty
        self.flush_every = max(1, flush_every)

        self._file: Optional[aiofiles.threadpool.binary.AsyncBufferedWriter] = None
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
    - links/metadata сохраняем как JSON-строки
    """

    def __init__(self, path: str, encoding: str = "utf-8", delimiter: str = ",", flush_every: int = 50):
        self.path = path
        self.encoding = encoding
        self.delimiter = delimiter
        self.flush_every = max(1, flush_every)

        self._file: Optional[aiofiles.threadpool.binary.AsyncBufferedWriter] = None
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

            if self._headers is None and (not Path(self.path).exists() or Path(self.path).stat().st_size == 0):
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
            elif self._headers is None:
                try:
                    async with aiofiles.open(self.path, "r", encoding=self.encoding, newline="") as rf:
                        first_line = await rf.readline()
                    if first_line:
                        reader = csv.reader([first_line], delimiter=self.delimiter)
                        self._headers = next(reader)
                    else:
                        self._headers = list(row.keys())
                except Exception:
                    self._headers = list(row.keys())

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
    SQLite + batch inserts
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
# Config
# =========================
@dataclass
class CrawlerConfig:
    # seed
    start_urls: List[str] = field(default_factory=list)
    sitemap_urls: List[str] = field(default_factory=list)

    # crawl limits
    max_pages: int = 100
    max_depth: int = 2
    max_concurrent: int = 10
    rate_limit: float = 0.0  # req/sec, 0 = unlimited

    # monitoring
    monitor_interval: float = 1.0  # seconds

    # http
    timeout_total: float = 15.0
    user_agent: str = "crawler_day7/1.0"

    # filters
    allowed_domains: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)  # substring match
    include_patterns: List[str] = field(default_factory=list)  # substring match

    # robots (stub)
    respect_robots: bool = False

    # storage
    storage: str = "ndjson"  # ndjson|csv|sqlite|none
    output: str = "results.ndjson"
    sqlite_table: str = "pages"

    # exports
    stats_json: str = "stats.json"
    report_html: str = "report.html"

    # logging
    log_file: str = "crawler.log"
    log_level: str = "INFO"

    # retries
    max_retries: int = 3
    backoff_factor: float = 2.0
    base_delay: float = 0.5

    @staticmethod
    def from_file(path: str) -> "CrawlerConfig":
        ext = Path(path).suffix.lower()
        raw = Path(path).read_text(encoding="utf-8")

        if ext in (".yaml", ".yml"):
            try:
                import yaml  # pip install pyyaml
            except Exception as e:
                raise RuntimeError("YAML config requires pyyaml (pip install pyyaml)") from e
            data = yaml.safe_load(raw) or {}
        elif ext == ".json":
            data = json.loads(raw)
        else:
            raise ValueError("Config must be .json or .yaml/.yml")

        return CrawlerConfig(**data)


# =========================
# CrawlerStats
# =========================
@dataclass
class CrawlerStats:
    started_at_perf: float = field(default_factory=time.perf_counter)
    finished_at_perf: Optional[float] = None

    started_at_dt: datetime = field(default_factory=utc_now)
    finished_at_dt: Optional[datetime] = None

    processed_pages: int = 0
    successful: int = 0
    failed: int = 0

    status_codes: Counter = field(default_factory=Counter)
    domains: Counter = field(default_factory=Counter)

    bytes_downloaded: int = 0

    def on_result(self, r: FetchResult) -> None:
        self.processed_pages += 1
        if r.ok:
            self.successful += 1
        else:
            self.failed += 1

        if r.status is not None:
            self.status_codes[str(r.status)] += 1

        d = urlparse(r.url).netloc.lower()
        if d:
            self.domains[d] += 1

        if r.text:
            self.bytes_downloaded += len(r.text.encode("utf-8", errors="ignore"))

    def finish(self) -> None:
        self.finished_at_perf = time.perf_counter()
        self.finished_at_dt = utc_now()

    @property
    def uptime_s(self) -> float:
        end = self.finished_at_perf or time.perf_counter()
        return max(0.0, end - self.started_at_perf)

    @property
    def avg_speed_pages_per_s(self) -> float:
        t = self.uptime_s
        return (self.processed_pages / t) if t > 0 else 0.0

    def get_stats(self, top_n_domains: int = 10) -> Dict[str, Any]:
        return {
            "processed_pages": self.processed_pages,
            "successful": self.successful,
            "failed": self.failed,
            "avg_speed_pages_per_s": self.avg_speed_pages_per_s,
            "uptime_s": self.uptime_s,
            "status_codes": dict(self.status_codes),
            "top_domains": self.domains.most_common(top_n_domains),
            "unique_domains": len(self.domains),
            "bytes_downloaded": self.bytes_downloaded,
            "started_at_iso": to_iso(self.started_at_dt),
            "finished_at_iso": to_iso(self.finished_at_dt) if self.finished_at_dt else None,
        }


# =========================
# Link extraction
# =========================
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def extract_links(html: str, base_url: str) -> List[str]:
    out: List[str] = []
    for m in _HREF_RE.finditer(html or ""):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        out.append(urljoin(base_url, href))
    return out


# =========================
# AsyncCrawler engine (Day6) - adapted to reuse session
# =========================
class AsyncCrawlerEngine:
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
        self.timeout = timeout or aiohttp.ClientTimeout(total=10, connect=3, sock_connect=3, sock_read=5)

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, int, str]:
        try:
            async with session.get(url) as resp:
                status = resp.status
                ct = resp.headers.get("content-type", "")

                if status == 404:
                    raise PermanentError() from HttpStatusError(404, url)
                if status in (401, 403):
                    raise PermanentError() from HttpStatusError(status, url)
                if status in (429, 503) or status >= 500:
                    raise TransientError() from HttpStatusError(status, url)

                text = await resp.text(errors="ignore")
                return text, status, ct

        except asyncio.TimeoutError as e:
            raise TransientError() from e
        except aiohttp.ClientError as e:
            raise NetworkError() from e

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> FetchResult:
        start = time.monotonic()
        try:
            (text, status, ct), attempts = await self.retry_strategy.execute_with_retry(
                self._fetch_url, session, url, url=url
            )
            elapsed = time.monotonic() - start
            if attempts > 1:
                self.retry_strategy.stats.successful_retries += 1
            return FetchResult(ok=True, url=url, status=status, text=text, content_type=ct, attempts=attempts, elapsed_s=elapsed)
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

    def build_record(self, r: FetchResult, links: List[str]) -> Dict[str, Any]:
        return {
            "url": r.url,
            "title": "",
            "text": r.text or "",
            "links": links,
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

    async def save_record(self, record: Dict[str, Any]) -> None:
        if self.storage is None:
            return
        await save_with_retry(self.storage, record, max_retries=self.save_retries)

    async def close(self) -> None:
        if self.storage is not None:
            await self.storage.close()


# =========================
# Sitemap
# =========================
class SitemapParser:
    def __init__(self, engine: AsyncCrawlerEngine, logger_: logging.Logger):
        self.engine = engine
        self.logger = logger_

    async def fetch_sitemap(self, session: aiohttp.ClientSession, sitemap_url: str) -> List[str]:
        urls: List[str] = []
        seen: Set[str] = set()

        async def _walk(sm_url: str) -> None:
            if sm_url in seen:
                return
            seen.add(sm_url)

            r = await self.engine.fetch(session, sm_url)
            if not r.ok or not r.text:
                self.logger.warning("sitemap fetch failed | url=%s | status=%s", sm_url, r.status)
                return

            try:
                root = ET.fromstring(r.text)
            except Exception as e:
                self.logger.warning("sitemap parse failed | url=%s | err=%s", sm_url, type(e).__name__)
                return

            tag = root.tag.lower()
            if "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag == "sitemapindex":
                for sm in root.findall(".//{*}sitemap/{*}loc"):
                    loc = (sm.text or "").strip()
                    if loc:
                        await _walk(loc)
            elif tag == "urlset":
                for loc_el in root.findall(".//{*}url/{*}loc"):
                    loc = (loc_el.text or "").strip()
                    if loc:
                        urls.append(loc)
            else:
                for loc_el in root.findall(".//{*}loc"):
                    loc = (loc_el.text or "").strip()
                    if loc and loc.startswith("http"):
                        urls.append(loc)

        await _walk(sitemap_url)
        return list(dict.fromkeys(urls))


# =========================
# Rate limiter
# =========================
class RateLimiter:
    def __init__(self, rate_limit: float):
        self.rate = float(rate_limit or 0.0)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self.rate <= 0:
            return
        min_interval = 1.0 / self.rate
        async with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < min_interval:
                await asyncio.sleep(min_interval - delta)
            self._last = time.monotonic()


# =========================
# AdvancedCrawler (integration)
# =========================
class AdvancedCrawler:
    def __init__(self, cfg: CrawlerConfig, engine: AsyncCrawlerEngine, logger_: logging.Logger):
        self.cfg = cfg
        self.engine = engine
        self.logger = logger_
        self.stats = CrawlerStats()

        self.queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        self.visited: Set[str] = set()

        self._stop_event = asyncio.Event()
        self._active = 0
        self._active_lock = asyncio.Lock()

        self._limiter = RateLimiter(cfg.rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

        # strict max_pages handling
        self._sched_lock = asyncio.Lock()
        self._scheduled_pages = 0  # how many URLs were accepted/scheduled (seed+discovered). Hard upper bound.

    @classmethod
    def from_config(cls, path: str) -> "AdvancedCrawler":
        cfg = CrawlerConfig.from_file(path)
        lg = setup_logging(cfg.log_file, cfg.log_level)

        retry = RetryStrategy(
            max_retries=cfg.max_retries,
            backoff_factor=cfg.backoff_factor,
            base_delay=cfg.base_delay,
        )

        storage_obj = None
        if cfg.storage == "ndjson":
            storage_obj = JSONStorage(cfg.output)
        elif cfg.storage == "csv":
            storage_obj = CSVStorage(cfg.output)
        elif cfg.storage == "sqlite":
            storage_obj = SQLiteStorage(cfg.output, table=cfg.sqlite_table)
        elif cfg.storage == "none":
            storage_obj = None
        else:
            raise ValueError(f"Unknown storage: {cfg.storage}")

        engine = AsyncCrawlerEngine(
            retry_strategy=retry,
            timeout=aiohttp.ClientTimeout(total=cfg.timeout_total),
            storage=storage_obj,
        )

        return cls(cfg, engine, lg)

    def _allowed(self, url: str) -> bool:
        try:
            u = urlparse(url)
        except Exception:
            return False

        if u.scheme not in ("http", "https"):
            return False
        if not u.netloc:
            return False

        if self.cfg.allowed_domains:
            d = u.netloc.lower()
            if not any(d == ad.lower() or d.endswith("." + ad.lower()) for ad in self.cfg.allowed_domains):
                return False

        s = url
        if self.cfg.include_patterns:
            if not any(p in s for p in self.cfg.include_patterns):
                return False
        if self.cfg.exclude_patterns:
            if any(p in s for p in self.cfg.exclude_patterns):
                return False

        return True

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        headers = {"User-Agent": self.cfg.user_agent}
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_total),
            headers=headers
        )
        return self._session

    async def _try_schedule(self, url: str, depth: int) -> bool:
        """
        Strictly schedules up to cfg.max_pages URLs total.
        Returns True if scheduled, False otherwise.
        """
        if not self._allowed(url):
            return False

        async with self._sched_lock:
            if self._scheduled_pages >= self.cfg.max_pages:
                return False
            if url in self.visited:
                return False
            self.visited.add(url)
            self._scheduled_pages += 1
            await self.queue.put((url, depth))
            return True

    async def _seed(self) -> None:
        if not self.cfg.allowed_domains and self.cfg.start_urls:
            self.cfg.allowed_domains = list({
                urlparse(u).netloc.lower()
                for u in self.cfg.start_urls
                if urlparse(u).netloc
            })
            self.logger.info("auto allowed_domains=%s", self.cfg.allowed_domains)

        for u in self.cfg.start_urls:
            await self._try_schedule(u, 0)

        if self.cfg.sitemap_urls:
            session = await self._ensure_session()
            smp = SitemapParser(self.engine, self.logger)
            for sm_url in self.cfg.sitemap_urls:
                urls = await smp.fetch_sitemap(session, sm_url)
                self.logger.info("sitemap loaded | %s | urls=%d", sm_url, len(urls))
                for u in urls:
                    ok = await self._try_schedule(u, 0)
                    if not ok:
                        break

    async def _inc_active(self, delta: int) -> None:
        async with self._active_lock:
            self._active += delta

    async def _worker(self, wid: int) -> None:
        session = await self._ensure_session()
        while not self._stop_event.is_set():
            # if already processed enough, stop consuming more work
            if self.stats.processed_pages >= self.cfg.max_pages:
                self._stop_event.set()
                break

            try:
                url, depth = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # stop when nothing queued and nobody active
                async with self._active_lock:
                    active = self._active
                if self.queue.empty() and active == 0:
                    self._stop_event.set()
                continue

            # HARD STOP: never fetch beyond max_pages
            if self.stats.processed_pages >= self.cfg.max_pages:
                self.queue.task_done()
                self._stop_event.set()
                continue

            await self._inc_active(1)
            try:
                await self._limiter.wait()
                r = await self.engine.fetch(session, url)

                if not r.ok:
                    self.logger.warning(
                        "fetch failed | url=%s | status=%s | error_type=%s",
                        url,
                        r.status,
                        r.error_type
                    )

                links: List[str] = []
                if r.ok and r.text and depth < self.cfg.max_depth:
                    links = extract_links(r.text, url)
                    for link in links:
                        if self._stop_event.is_set():
                            break
                        # schedule strictly by max_pages
                        ok = await self._try_schedule(link, depth + 1)
                        if not ok:
                            # If can't schedule because limit reached, stop early
                            async with self._sched_lock:
                                if self._scheduled_pages >= self.cfg.max_pages:
                                    break

                # count AFTER fetch (processed pages)
                self.stats.on_result(r)

                # if we hit limit, stop event now
                if self.stats.processed_pages >= self.cfg.max_pages:
                    self._stop_event.set()

                # store record (even failed, for audit)
                record = self.engine.build_record(r, links)
                try:
                    await self.engine.save_record(record)
                except Exception:
                    self.logger.exception("storage save failed | url=%s", url)

            finally:
                self.queue.task_done()
                await self._inc_active(-1)

    async def _monitor_loop(self) -> None:
        interval = float(getattr(self.cfg, "monitor_interval", 1.0) or 1.0)

        last_processed = 0
        last_t = time.monotonic()
        last_log_t = last_t
        HEARTBEAT_SEC = interval * 3

        while not self._stop_event.is_set():
            await asyncio.sleep(interval)

            processed = self.stats.processed_pages
            now = time.monotonic()
            dt = max(1e-9, now - last_t)

            delta = processed - last_processed
            heartbeat = (now - last_log_t) >= HEARTBEAT_SEC

            if delta <= 0 and not heartbeat:
                continue

            speed = delta / dt if delta > 0 else 0.0
            remaining = max(0, self.cfg.max_pages - processed)
            eta_s = (remaining / speed) if speed > 0 else float("inf")

            async with self._active_lock:
                active = self._active

            self.logger.info(
                "progress | processed=%d/%d | queued=%d | active=%d | scheduled=%d/%d | speed=%.2f p/s | eta=%s",
                processed,
                self.cfg.max_pages,
                self.queue.qsize(),
                active,
                self._scheduled_pages,
                self.cfg.max_pages,
                speed,
                f"{eta_s:.0f}s" if eta_s != float("inf") else "∞",
            )

            last_processed = processed
            last_t = now
            last_log_t = now

    async def crawl(self) -> None:
        await self._seed()
        if self.queue.empty():
            self.logger.warning("no start urls to crawl")
            self.stats.finish()
            self._stop_event.set()
            return

        monitor_task = asyncio.create_task(self._monitor_loop())
        workers = [asyncio.create_task(self._worker(i)) for i in range(self.cfg.max_concurrent)]

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.2)

                if self.stats.processed_pages >= self.cfg.max_pages:
                    self._stop_event.set()
                    break

                async with self._active_lock:
                    active = self._active
                if self.queue.empty() and active == 0:
                    self._stop_event.set()
                    break

            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        finally:
            self.stats.finish()
            self._stop_event.set()

            # monitor: do not silently swallow exceptions
            if not monitor_task.done():
                monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("monitor failed")

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.get_stats()

    def export_to_json(self, filename: str) -> None:
        Path(filename).write_text(json.dumps(self.get_stats(), ensure_ascii=False, indent=2), encoding="utf-8")

    def export_to_html_report(self, filename: str) -> None:
        stats = self.get_stats()

        status_rows = "".join(
            f"<tr><td>{html_escape(str(code))}</td><td>{int(cnt)}</td></tr>"
            for code, cnt in sorted(stats["status_codes"].items(), key=lambda x: int(x[0]))
        )

        dom_rows = "".join(
            f"<tr><td>{html_escape(d)}</td><td>{int(cnt)}</td></tr>"
            for d, cnt in stats["top_domains"]
        )

        def bar_chart(items: List[Tuple[str, int]], *, max_bars: int = 12) -> str:
            items = items[:max_bars]
            if not items:
                return "<div class='muted'>no data</div>"
            m = max(v for _, v in items) or 1
            bars = []
            for label, val in items:
                w = int((val / m) * 100)
                bars.append(
                    f"""
                    <div class="barrow">
                      <div class="barlabel">{html_escape(label)}</div>
                      <div class="barwrap">
                        <div class="bar" style="width:{w}%"></div>
                      </div>
                      <div class="barval">{val}</div>
                    </div>
                    """
                )
            return "\n".join(bars)

        status_for_chart = sorted(
            [(str(k), int(v)) for k, v in stats["status_codes"].items()],
            key=lambda x: x[0]
        )
        domains_for_chart = [(d, int(cnt)) for d, cnt in stats["top_domains"]]

        html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Crawler report</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #121826;
      --text: #e6eaf2;
      --muted: #9aa4b2;
      --border: #263044;
      --bar: #4f8cff;
      --ok: #34d399;
      --bad: #fb7185;
    }}
    body {{
      margin: 0; padding: 24px;
      background: var(--bg); color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
    h1 {{ margin: 0 0 16px; font-size: 20px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
      max-width: 1100px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-12 {{ grid-column: span 12; }}
    .kpi {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 12px;
    }}
    .kpi .label {{ color: var(--muted); }}
    .kpi .value {{ font-weight: 700; }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 10px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}

    .barrow {{
      display: grid;
      grid-template-columns: 140px 1fr 44px;
      gap: 10px;
      align-items: center;
      margin: 6px 0;
    }}
    .barlabel {{
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .barwrap {{
      height: 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.03);
    }}
    .bar {{ height: 100%; background: var(--bar); }}
    .barval {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}

    .muted {{ color: var(--muted); }}
    .row {{
      display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
      margin-top: 6px;
    }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    @media (max-width: 900px) {{
      .span-4, .span-6 {{ grid-column: span 12; }}
      .barrow {{ grid-template-columns: 120px 1fr 44px; }}
    }}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card span-12">
      <h1>AdvancedCrawler report</h1>
      <div class="row">
        <span class="pill">uptime: {stats.get("uptime_s",0):.2f}s</span>
        <span class="pill">avg speed: {stats.get("avg_speed_pages_per_s",0):.2f} pages/s</span>
        <span class="pill">processed: {stats.get("processed_pages",0)}</span>
        <span class="pill">ok: <span class="ok">{stats.get("successful",0)}</span></span>
        <span class="pill">failed: <span class="bad">{stats.get("failed",0)}</span></span>
      </div>
    </div>

    <div class="card span-4">
      <h2>KPIs</h2>
      <div class="kpi">
        <div class="label">processed</div><div class="value">{stats.get("processed_pages",0)}</div>
        <div class="label">successful</div><div class="value ok">{stats.get("successful",0)}</div>
        <div class="label">failed</div><div class="value bad">{stats.get("failed",0)}</div>
        <div class="label">unique domains</div><div class="value">{stats.get("unique_domains",0)}</div>
      </div>
    </div>

    <div class="card span-4">
      <h2>Status codes (bars)</h2>
      {bar_chart(status_for_chart, max_bars=12)}
    </div>

    <div class="card span-4">
      <h2>Top domains (bars)</h2>
      {bar_chart(domains_for_chart, max_bars=12)}
    </div>

    <div class="card span-6">
      <h2>Status codes (table)</h2>
      <table>
        <thead><tr><th>Status</th><th>Count</th></tr></thead>
        <tbody>{status_rows or "<tr><td colspan='2' class='muted'>no data</td></tr>"}</tbody>
      </table>
    </div>

    <div class="card span-6">
      <h2>Top domains (table)</h2>
      <table>
        <thead><tr><th>Domain</th><th>Count</th></tr></thead>
        <tbody>{dom_rows or "<tr><td colspan='2' class='muted'>no data</td></tr>"}</tbody>
      </table>
    </div>

    <div class="card span-12">
      <h2>Raw stats (JSON)</h2>
      <pre style="white-space:pre-wrap; word-break:break-word; margin:0; color: var(--muted);">{html_escape(json.dumps(stats, ensure_ascii=False, indent=2))}</pre>
    </div>
  </div>
</body>
</html>
"""
        Path(filename).write_text(html, encoding="utf-8")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self.engine.close()


def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


async def demo_from_config(config_path: str = "config.yaml") -> None:
    crawler = AdvancedCrawler.from_config(config_path)
    try:
        await crawler.crawl()

        stats = crawler.get_stats()
        print(f"Обработано: {stats.get('processed_pages', 0)} страниц")
        print(f"Успешно: {stats.get('successful', 0)}")
        print(f"Ошибок: {stats.get('failed', 0)}")

        crawler.export_to_html_report("report.html")
    finally:
        await crawler.close()


# =========================
# CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crawler_day7.py", description="Advanced async crawler (Day7)")
    p.add_argument("--urls", nargs="*", default=None, help="Start URLs (space-separated)")
    p.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl")
    p.add_argument("--max-depth", type=int, default=None, help="Maximum crawl depth")
    p.add_argument("--output", type=str, default=None, help="Output storage file (ndjson/csv/sqlite)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")
    p.add_argument("--respect-robots", action="store_true", help="Respect robots.txt (stub)")
    p.add_argument("--rate-limit", type=float, default=None, help="Requests per second (0 = unlimited)")
    p.add_argument("--log-level", type=str, default=None, help="DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--log-file", type=str, default=None, help="Log file path")
    p.add_argument("--storage", type=str, default=None, help="ndjson|csv|sqlite|none")
    p.add_argument("--sitemap", nargs="*", default=None, help="Sitemap URLs (optional)")
    p.add_argument("--stats-json", type=str, default=None, help="Stats JSON output file")
    p.add_argument("--report-html", type=str, default=None, help="HTML report output file")
    p.add_argument("--demo", action="store_true", help="Run demo: AdvancedCrawler.from_config(config.yaml)")
    p.add_argument("--allowed-domains", nargs="*", default=None, help="Allowed domains (scope)")
    p.add_argument("--monitor-interval", type=float, default=None, help="Monitor interval seconds")
    return p


async def async_main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.demo:
        await demo_from_config(args.config or "config.yaml")
        return 0

    if args.config:
        cfg = CrawlerConfig.from_file(args.config)
    else:
        cfg = CrawlerConfig()

    # CLI overrides
    if args.urls is not None and len(args.urls) > 0:
        cfg.start_urls = args.urls
    if args.sitemap is not None and len(args.sitemap) > 0:
        cfg.sitemap_urls = args.sitemap
    if args.max_pages is not None:
        cfg.max_pages = args.max_pages
    if args.max_depth is not None:
        cfg.max_depth = args.max_depth
    if args.output is not None:
        cfg.output = args.output
    if args.storage is not None:
        cfg.storage = args.storage
    if args.rate_limit is not None:
        cfg.rate_limit = args.rate_limit
    if args.respect_robots:
        cfg.respect_robots = True
    if args.log_level is not None:
        cfg.log_level = args.log_level
    if args.log_file is not None:
        cfg.log_file = args.log_file
    if args.stats_json is not None:
        cfg.stats_json = args.stats_json
    if args.report_html is not None:
        cfg.report_html = args.report_html
    if args.allowed_domains is not None and len(args.allowed_domains) > 0:
        cfg.allowed_domains = args.allowed_domains
    if args.monitor_interval is not None:
        cfg.monitor_interval = args.monitor_interval

    global logger
    logger = setup_logging(cfg.log_file, cfg.log_level)

    retry = RetryStrategy(max_retries=cfg.max_retries, backoff_factor=cfg.backoff_factor, base_delay=cfg.base_delay)

    storage: Optional[DataStorage] = None
    st = cfg.storage.lower()
    if st == "ndjson":
        storage = JSONStorage(cfg.output)
    elif st == "csv":
        storage = CSVStorage(cfg.output)
    elif st == "sqlite":
        storage = SQLiteStorage(cfg.output, table=cfg.sqlite_table)
    elif st == "none":
        storage = None
    else:
        raise SystemExit("storage must be ndjson|csv|sqlite|none")

    engine = AsyncCrawlerEngine(
        retry_strategy=retry,
        timeout=aiohttp.ClientTimeout(total=cfg.timeout_total),
        storage=storage,
        save_retries=3,
    )

    crawler = AdvancedCrawler(cfg, engine, logger)

    logger.info(
        "start | urls=%d | sitemaps=%d | max_pages=%d | depth=%d | concurrent=%d | rate=%.2f | monitor=%.2fs",
        len(cfg.start_urls),
        len(cfg.sitemap_urls),
        cfg.max_pages,
        cfg.max_depth,
        cfg.max_concurrent,
        cfg.rate_limit,
        cfg.monitor_interval,
    )

    try:
        await crawler.crawl()

        crawler.export_to_json(cfg.stats_json)
        crawler.export_to_html_report(cfg.report_html)

        logger.info(
            "done | processed=%d | ok=%d | failed=%d | avg_speed=%.2f p/s",
            crawler.stats.processed_pages,
            crawler.stats.successful,
            crawler.stats.failed,
            crawler.stats.avg_speed_pages_per_s,
        )
        return 0
    finally:
        await crawler.close()


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
