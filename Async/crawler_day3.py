import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import aiohttp


# ---------- logging ----------
logger = logging.getLogger("crawler_day3")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------- models ----------
@dataclass
class FetchResult:
    ok: bool
    status: Optional[int]
    text: Optional[str]
    error: Optional[str]
    final_url: Optional[str] = None
    content_type: Optional[str] = None


# ---------- queue ----------

class CrawlerQueue:
    def __init__(self) -> None:
        self._pq: asyncio.PriorityQueue[Tuple[int, int, str]] = asyncio.PriorityQueue()
        self._counter = 0

        # "навсегда видели" (ключевой фикс)
        self._scheduled: Set[str] = set()

        # чисто для статистики "что прямо сейчас лежит в очереди"
        self._queued_set: Set[str] = set()

        self._processed: Dict[str, dict] = {}
        self._failed: Dict[str, str] = {}

        self._lock = asyncio.Lock()

    def _norm(self, url: str) -> str:
        url = url.strip()
        url, _frag = urldefrag(url)
        return url

    def add_url(self, url: str, priority: int = 0) -> bool:
        url = self._norm(url)
        if not url:
            return False

        # 👇 ВАЖНО: дедуп по scheduled, а не по queued_set
        if url in self._scheduled or url in self._processed or url in self._failed:
            return False

        self._scheduled.add(url)
        self._queued_set.add(url)

        self._counter += 1
        self._pq.put_nowait((-priority, self._counter, url))
        return True

    async def get_next(self) -> str | None:
        try:
            _neg_prio, _cnt, url = await asyncio.wait_for(self._pq.get(), timeout=0.2)
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            self._queued_set.discard(url)
        return url

    def mark_processed(self, url: str) -> None:
        url = self._norm(url)
        self._processed[url] = {"ok": True}

    def mark_failed(self, url: str, error: str) -> None:
        url = self._norm(url)
        self._failed[url] = error

    def get_stats(self) -> dict:
        return {
            "queued": self._pq.qsize(),
            "scheduled": len(self._scheduled),
            "processed": len(self._processed),
            "failed": len(self._failed),
        }


    @property
    def processed(self) -> Dict[str, dict]:
        return self._processed

    @property
    def failed(self) -> Dict[str, str]:
        return self._failed


# ---------- semaphores ----------
class _SemLease:
    def __init__(self, mgr: "SemaphoreManager", domain: str):
        self.mgr = mgr
        self.domain = domain

    async def __aenter__(self):
        await self.mgr._acquire(self.domain)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.mgr._release(self.domain)
        return False


class SemaphoreManager:
    """
    - глобальный лимит конкурентности
    - лимит конкурентности на домен
    - активные задачи (global и per-domain)
    """

    def __init__(self, max_global: int = 10, max_per_domain: int = 2):
        self.max_global = max_global
        self.max_per_domain = max_per_domain

        self._global = asyncio.Semaphore(max_global)
        self._domain_sems: Dict[str, asyncio.Semaphore] = {}

        self._active_global = 0
        self._active_by_domain: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    def for_url(self, url: str) -> _SemLease:
        domain = urlparse(url).netloc.lower()
        return _SemLease(self, domain)

    def _get_domain_sem(self, domain: str) -> asyncio.Semaphore:
        sem = self._domain_sems.get(domain)
        if sem is None:
            sem = asyncio.Semaphore(self.max_per_domain)
            self._domain_sems[domain] = sem
        return sem

    async def _acquire(self, domain: str) -> None:
        dom_sem = self._get_domain_sem(domain)
        await self._global.acquire()
        await dom_sem.acquire()
        async with self._lock:
            self._active_global += 1
            self._active_by_domain[domain] = self._active_by_domain.get(domain, 0) + 1

    async def _release(self, domain: str) -> None:
        dom_sem = self._get_domain_sem(domain)
        dom_sem.release()
        self._global.release()
        async with self._lock:
            self._active_global = max(0, self._active_global - 1)
            self._active_by_domain[domain] = max(0, self._active_by_domain.get(domain, 0) - 1)

    def get_active_stats(self) -> dict:
        return {
            "active_global": self._active_global,
            "active_by_domain": dict(self._active_by_domain),
            "max_global": self.max_global,
            "max_per_domain": self.max_per_domain,
        }


# ---------- crawler ----------
class AsyncCrawler:
    """
    День 3 — версия строго по ТЗ:

    - crawl(self, start_urls: list[str], max_pages: int = 100)
    - очередь + visited/failed/processed
    - контроль глубины (max_depth)
    - фильтры URL:
        - same_domain_only
        - exclude_patterns
        - include_patterns
    - семафоры: global + per-domain
    - прогресс
    """

    HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

    def __init__(
        self,
        max_concurrent: int = 10,
        max_depth: int = 2,
        max_per_domain: int = 2,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        user_agent: str = "crawler_day3/1.0",
        # Настройки фильтрации (по ТЗ они есть, но не обязаны быть параметрами crawl)
        same_domain_only: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        # для тестов (инъекция)
        fetch_override=None,
        extract_override=None,
    ):
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []
        

        self.queue = CrawlerQueue()
        self.sems = SemaphoreManager(max_global=max_concurrent, max_per_domain=max_per_domain)
        self.seen_urls: Set[str] = set()

        # state
        self.visited_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}
        self.processed_urls: Dict[str, dict] = {}
        self.url_depth: Dict[str, int] = {}

        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._user_agent = user_agent

        self._fetch_override = fetch_override
        self._extract_override = extract_override

        self._start_time = 0.0
        self._processed_count = 0
        self._stop = asyncio.Event()

    def _norm(self, url: str) -> str:
        url = url.strip()
        url, _frag = urldefrag(url)
        return url

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _passes_filters(self, url: str, base_domain: Optional[str]) -> bool:
        if not url:
            return False

        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme not in ("http", "https"):
            return False

        if self.same_domain_only and base_domain:
            if self._domain(url) != base_domain:
                return False

        for p in self.exclude_patterns:
            if re.search(p, url):
                return False

        if self.include_patterns:
            ok = any(re.search(p, url) for p in self.include_patterns)
            if not ok:
                return False

        return True

    def extract_links(self, html: str, base_url: str) -> List[str]:
        if self._extract_override:
            return self._extract_override(html, base_url)

        links: List[str] = []
        for m in self.HREF_RE.finditer(html or ""):
            href = m.group(1).strip()
            if not href or href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            abs_url = urljoin(base_url, href)
            abs_url = self._norm(abs_url)
            links.append(abs_url)
        return links

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> FetchResult:
        if self._fetch_override:
            return await self._fetch_override(session, url)

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self._connect_timeout, sock_read=self._read_timeout)
        headers = {"User-Agent": self._user_agent}

        try:
            async with session.get(url, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                ct = resp.headers.get("Content-Type", "")
                text = await resp.text(errors="ignore")
                ok = 200 <= resp.status < 300
                return FetchResult(
                    ok=ok,
                    status=resp.status,
                    text=text,
                    error=None if ok else f"HTTP {resp.status}",
                    final_url=str(resp.url),
                    content_type=ct,
                )
        except asyncio.TimeoutError:
            return FetchResult(ok=False, status=None, text=None, error="Timeout")
        except aiohttp.ClientError as e:
            return FetchResult(ok=False, status=None, text=None, error=f"ClientError: {e}")
        except Exception as e:
            return FetchResult(ok=False, status=None, text=None, error=f"Error: {e}")

    # ---------- MAIN API (по ТЗ) ----------
    async def crawl(self, start_urls: list[str], max_pages: int = 100) -> Dict[str, dict]:
        self._start_time = time.time()
        self._processed_count = 0
        self._stop.clear() 

        # базовый домен — от первого стартового URL
        base_domain = self._domain(start_urls[0]) if (self.same_domain_only and start_urls) else None

        # seed
        for u in start_urls:
            u = self._norm(u)
            if self._passes_filters(u, base_domain):
                if u not in self.seen_urls:
                    self.seen_urls.add(u)
                    self.url_depth[u] = 0
                    self.queue.add_url(u, priority=10)

        async with aiohttp.ClientSession() as session:
            progress_task = asyncio.create_task(self._progress_loop(max_pages=max_pages))

            workers = [
                asyncio.create_task(self._worker(session=session, base_domain=base_domain, max_pages=max_pages))
                for _ in range(self.sems.max_global)
            ]

            await asyncio.gather(*workers)

            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

        return dict(self.processed_urls)

    async def _worker(self, session: aiohttp.ClientSession, base_domain: Optional[str], max_pages: int):
        while True:
            if self._stop.is_set():
                return

            url = await self.queue.get_next()
            if url is None:
                await asyncio.sleep(0.05)
                if self.queue.get_stats()["queued"] == 0:
                    return
                continue

            url = self._norm(url)

            if url in self.visited_urls:
                continue

            depth = self.url_depth.get(url, 0)
            if depth > self.max_depth:
                continue

            self.visited_urls.add(url)

            async with self.sems.for_url(url):
                fr = await self.fetch(session, url)

            if not fr.ok or fr.text is None:
                err = fr.error or "Unknown error"
                self.failed_urls[url] = err
                self.queue.mark_failed(url, err)
                continue

            links = self.extract_links(fr.text, fr.final_url or url)
            data = {
                "url": url,
                "final_url": fr.final_url or url,
                "status": fr.status,
                "content_type": fr.content_type,
                "bytes": len(fr.text.encode("utf-8", errors="ignore")),
                "depth": depth,
                "links_count": len(links),
                "links": links,
            }

            self.processed_urls[url] = data
            self.queue.mark_processed(url)

            self._processed_count += 1
            if self._processed_count >= max_pages:
                self._stop.set()
                return

            # добавить ссылки глубже
            if self._stop.is_set():
                return

            next_depth = depth + 1
            if next_depth <= self.max_depth:
                for link in links:
                    link = self._norm(link)
                    if not self._passes_filters(link, base_domain):
                        continue

                    # режем дубли на входе
                    if link in self.seen_urls:
                        continue
                    self.seen_urls.add(link)

                    self.url_depth[link] = next_depth
                    self.queue.add_url(link, priority=max(0, 5 - next_depth))
            
    async def _progress_loop(self, max_pages: int):
        last = time.time()
        last_count = 0

        while True:
            now = time.time()
            dt = max(1e-9, now - last)

            processed = len(self.processed_urls)
            failed = len(self.failed_urls)
            stats = self.queue.get_stats()
            queued = stats["queued"]
            scheduled = stats.get("scheduled", 0)

            speed = (processed - last_count) / dt
            total_speed = processed / max(1e-9, now - self._start_time)

            logger.info(
                "progress | processed=%d/%d | queued=%d | scheduled=%d | failed=%d | +%.2f p/s | avg=%.2f p/s",
                processed,
                max_pages,
                queued,
                scheduled,
                failed,
                speed,
                total_speed,
            )

            last = now
            last_count = processed

            await asyncio.sleep(1.0)

# ---------- demo ----------
async def main():
    crawler = AsyncCrawler(
        max_concurrent=10,
        max_depth=2,
        max_per_domain=2,
        same_domain_only=True,
        exclude_patterns=[r"\.pdf$", r"\.jpg$", r"\.png$"],
    )
    results = await crawler.crawl(start_urls=["https://books.toscrape.com/"], max_pages=50)
    print(f"Обработано: {len(results)} страниц")

    with open("crawl_results.jsonl", "w", encoding="utf-8") as f:
        for _url, data in results.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print("Сохранено: crawl_results.jsonl")


if __name__ == "__main__":
    print("RUN DEMO: starting...")
    asyncio.run(main())
    print("RUN DEMO: finished.")