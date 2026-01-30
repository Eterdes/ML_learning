# crawler_day4.py
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import aiohttp
from aiohttp import web
from aiohttp.client_exceptions import ClientError


# ---------------- logging ----------------
logger = logging.getLogger("crawler_day4")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------- utils ----------------
def _domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def _base_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


# ---------------- RateLimiter ----------------
class RateLimiter:
    """
    Простая "gap-based" реализация:
    requests_per_second => минимальный интервал = 1/rps
    - per_domain=True: отдельный интервал на домен
    - per_domain=False: общий глобальный интервал
    """

    def __init__(self, requests_per_second: float = 1.0, per_domain: bool = True):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self.rps = float(requests_per_second)
        self.per_domain = bool(per_domain)
        self._interval = 1.0 / self.rps

        self._lock = asyncio.Lock()
        self._next_allowed_global: float = 0.0
        self._next_allowed_by_domain: Dict[str, float] = {}

    async def acquire(self, domain: str = None) -> float:
        """
        Ждём до разрешённого времени.
        Возвращаем фактическое ожидание (seconds).
        """
        key = (domain or "").lower() if self.per_domain else None

        async with self._lock:
            now = time.monotonic()
            if self.per_domain:
                next_allowed = self._next_allowed_by_domain.get(key, 0.0)
            else:
                next_allowed = self._next_allowed_global

            wait_for = max(0.0, next_allowed - now)

            # бронируем следующий слот заранее (чтобы параллельные acquire не "схлопнулись")
            reserved_time = max(next_allowed, now) + self._interval
            if self.per_domain:
                self._next_allowed_by_domain[key] = reserved_time
            else:
                self._next_allowed_global = reserved_time

        if wait_for > 0:
            await asyncio.sleep(wait_for)
        return wait_for


# ---------------- RobotsParser ----------------
@dataclass
class RobotsRules:
    fetched_at: float
    disallow_by_agent: Dict[str, List[str]] = field(default_factory=dict)
    crawl_delay_by_agent: Dict[str, float] = field(default_factory=dict)


class RobotsParser:
    """
    Мини-парсер robots.txt:
    - User-agent, Disallow, Crawl-delay
    - Кэш по домену
    - ТЗ: get_crawl_delay(user_agent="*") -> float (для последнего fetch_robots)
    """

    def __init__(self, session: aiohttp.ClientSession, cache_ttl: float = 3600.0):
        self.session = session
        self.cache_ttl = float(cache_ttl)
        self._cache: Dict[str, RobotsRules] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        # важно для ТЗ-сигнатуры get_crawl_delay(...)
        self._current_domain: Optional[str] = None

    async def fetch_robots(self, base_url: str) -> dict:
        domain = _domain_of(base_url)
        self._current_domain = domain  # <-- запоминаем “текущий” домен

        now = time.monotonic()
        cached = self._cache.get(domain)
        if cached and (now - cached.fetched_at) < self.cache_ttl:
            return {"domain": domain, "cached": True}

        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            now2 = time.monotonic()
            cached2 = self._cache.get(domain)
            if cached2 and (now2 - cached2.fetched_at) < self.cache_ttl:
                return {"domain": domain, "cached": True}

            robots_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", "robots.txt")
            text = ""
            try:
                async with self.session.get(robots_url) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="ignore")
                    else:
                        text = ""
            except Exception:
                text = ""

            disallow_by_agent, crawl_delay_by_agent = self._parse_robots(text)
            self._cache[domain] = RobotsRules(
                fetched_at=time.monotonic(),
                disallow_by_agent=disallow_by_agent,
                crawl_delay_by_agent=crawl_delay_by_agent,
            )
            return {"domain": domain, "cached": False, "robots_url": robots_url}

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        domain = _domain_of(url)
        path = urlparse(url).path or "/"

        rules = self._cache.get(domain)
        if not rules:
            return True

        ua = (user_agent or "*").lower()
        disallow = self._select_list_for_agent(rules.disallow_by_agent, ua)

        for prefix in disallow:
            if not prefix:
                continue
            if path.startswith(prefix):
                return False
        return True

    # ---- ТЗ-метод: без url/base_url ----
    def get_crawl_delay(self, user_agent: str = "*") -> float:
        """
        Возвращает Crawl-delay из robots.txt для домена,
        который последний раз был загружен через fetch_robots(...).
        Если fetch_robots ещё не вызывали — 0.0
        """
        if not self._current_domain:
            return 0.0
        rules = self._cache.get(self._current_domain)
        if not rules:
            return 0.0

        ua = (user_agent or "*").lower()
        return float(self._select_delay_for_agent(rules.crawl_delay_by_agent, ua))

    # ---- удобный метод (не обязателен по ТЗ), если хочешь ----
    def get_crawl_delay_for(self, base_url_or_url: str, user_agent: str = "*") -> float:
        domain = _domain_of(base_url_or_url)
        rules = self._cache.get(domain)
        if not rules:
            return 0.0
        ua = (user_agent or "*").lower()
        return float(self._select_delay_for_agent(rules.crawl_delay_by_agent, ua))

    @staticmethod
    def _select_list_for_agent(mapping: Dict[str, List[str]], ua: str) -> List[str]:
        if ua in mapping:
            return mapping[ua]
        if "*" in mapping:
            return mapping["*"]
        return []

    @staticmethod
    def _select_delay_for_agent(mapping: Dict[str, float], ua: str) -> float:
        if ua in mapping:
            return mapping[ua]
        if "*" in mapping:
            return mapping["*"]
        return 0.0

    @staticmethod
    def _parse_robots(text: str) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        disallow_by_agent: Dict[str, List[str]] = {}
        crawl_delay_by_agent: Dict[str, float] = {}

        current_agents: List[str] = []
        for raw in (text or "").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue

            k, v = [x.strip() for x in line.split(":", 1)]
            k = k.lower()
            v = v.strip()

            if k == "user-agent":
                agent = (v or "*").lower()
                current_agents = [agent]
                disallow_by_agent.setdefault(agent, [])
            elif k == "disallow":
                for a in (current_agents or ["*"]):
                    disallow_by_agent.setdefault(a, []).append(v)
            elif k == "crawl-delay":
                try:
                    delay = float(v)
                except ValueError:
                    continue
                for a in (current_agents or ["*"]):
                    crawl_delay_by_agent[a] = delay

        return disallow_by_agent, crawl_delay_by_agent


# ---------------- crawler ----------------
@dataclass
class FetchResult:
    ok: bool
    status: Optional[int]
    text: Optional[str]
    error: Optional[str]
    url: str
    waited: float = 0.0
    blocked_by_robots: bool = False


class AsyncCrawler:
    def __init__(
        self,
        max_concurrent: int = 10,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        requests_per_second: float = 1.0,
        per_domain: bool = True,
        respect_robots: bool = True,
        min_delay: float = 0.0,
        jitter: float = 0.0,
        backoff_base: float = 0.5,
        backoff_cap: float = 10.0,
        user_agent: str = "MyBot/1.0",
        user_agents: Optional[List[str]] = None,  # опциональная ротация
    ):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout, sock_read=read_timeout)

        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second, per_domain=per_domain)
        self.respect_robots = respect_robots
        self.min_delay = float(min_delay)
        self.jitter = float(jitter)

        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self._backoff_by_domain: Dict[str, int] = {}  # consecutive errors

        self._user_agent = user_agent
        self._user_agents = [ua for ua in (user_agents or []) if ua] or None
        self._ua_idx = 0

        # stats
        self._lock_stats = asyncio.Lock()
        self._req_timestamps: List[float] = []
        self._wait_samples: List[float] = []
        self.blocked_robots_count: int = 0

        self.session: Optional[aiohttp.ClientSession] = None
        self.robots: Optional[RobotsParser] = None

    def _pick_user_agent(self) -> str:
        if not self._user_agents:
            return self._user_agent
        ua = self._user_agents[self._ua_idx % len(self._user_agents)]
        self._ua_idx += 1
        return ua

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        self.robots = RobotsParser(self.session)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def fetch(self, url: str) -> FetchResult:
        if not self.session:
            raise RuntimeError("Use AsyncCrawler as an async context manager: async with AsyncCrawler(...) as c:")

        domain = _domain_of(url)
        base = _base_url(url)
        ua = self._pick_user_agent()

        # robots
        if self.respect_robots and self.robots:
            await self.robots.fetch_robots(base)
            if not self.robots.can_fetch(url, user_agent=ua):
                async with self._lock_stats:
                    self.blocked_robots_count += 1
                logger.info(f"robots BLOCK | {url}")
                return FetchResult(
                    ok=False,
                    status=None,
                    text=None,
                    error="blocked_by_robots",
                    url=url,
                    blocked_by_robots=True,
                )

        # delays
        waited_rate = await self.rate_limiter.acquire(domain)
        waited_delay = await self._apply_politeness_delay(domain, base, ua)
        waited = waited_rate + waited_delay

        async with self.sem:
            try:
                async with self.session.get(url, headers={"User-Agent": ua}) as resp:
                    text = await resp.text(errors="ignore")
                    ok = 200 <= resp.status < 300
                    if ok:
                        self._backoff_by_domain[domain] = 0
                    else:
                        self._inc_backoff(domain, resp.status)
                    await self._record_stats(waited)
                    return FetchResult(ok=ok, status=resp.status, text=text, error=None if ok else f"http_{resp.status}", url=url, waited=waited)
            except (asyncio.TimeoutError, ClientError) as e:
                self._inc_backoff(domain, None)
                await self._record_stats(waited)
                return FetchResult(ok=False, status=None, text=None, error=type(e).__name__, url=url, waited=waited)

    async def _apply_politeness_delay(self, domain: str, base_url: str, ua: str) -> float:
        # crawl-delay из robots
        crawl_delay = 0.0
        if self.respect_robots and self.robots:
            crawl_delay = self.robots.get_crawl_delay(user_agent=ua) or 0.0

        # backoff
        backoff_s = self._current_backoff_seconds(domain)

        # min_delay + jitter
        extra = self.min_delay
        if self.jitter > 0:
            extra += random.uniform(0.0, self.jitter)

        delay = max(crawl_delay, extra) + backoff_s
        if delay > 0:
            await asyncio.sleep(delay)
        return delay

    def _inc_backoff(self, domain: str, status: Optional[int]):
        # backoff только для "ошибок" (timeouts/client errors уже сюда попадают),
        # и для http >= 500 (а 4xx обычно не надо раздувать)
        n = self._backoff_by_domain.get(domain, 0)
        if status is None or (status >= 500):
            self._backoff_by_domain[domain] = min(n + 1, 30)
        else:
            self._backoff_by_domain[domain] = 0

    def _current_backoff_seconds(self, domain: str) -> float:
        n = self._backoff_by_domain.get(domain, 0)
        if n <= 0:
            return 0.0
        s = self.backoff_base * (2 ** (n - 1))
        return min(s, self.backoff_cap)

    async def _record_stats(self, waited: float):
        async with self._lock_stats:
            now = time.monotonic()
            self._req_timestamps.append(now)
            self._wait_samples.append(waited)
            # чистим старые точки > 10s (для "текущей" скорости)
            cutoff = now - 10.0
            while self._req_timestamps and self._req_timestamps[0] < cutoff:
                self._req_timestamps.pop(0)

    async def get_stats(self) -> dict:
        async with self._lock_stats:
            now = time.monotonic()
            # текущая скорость: кол-во за последние 10 секунд / 10
            rps = (len(self._req_timestamps) / 10.0) if self._req_timestamps else 0.0
            avg_wait = (sum(self._wait_samples) / len(self._wait_samples)) if self._wait_samples else 0.0
            return {
                "current_rps_10s": rps,
                "avg_total_wait_s": avg_wait,
                "robots_blocked": self.blocked_robots_count,
                "samples": len(self._wait_samples),
                "now_monotonic": now,
            }


# ---------------- demo (local site) ----------------
async def _demo_server() -> Tuple[web.AppRunner, str]:
    """
    Локальный сервер:
    - /robots.txt: Disallow /private и Crawl-delay
    - /, /page1, /page2, /private/secret
    """
    async def robots_txt(request):
        txt = "\n".join(
            [
                "User-agent: *",
                "Disallow: /private",
                "Crawl-delay: 1",
                "",
            ]
        )
        return web.Response(text=txt)

    async def ok_page(request):
        return web.Response(text=f"OK {request.path}")

    async def private_page(request):
        return web.Response(text="SECRET", status=200)

    app = web.Application()
    app.router.add_get("/robots.txt", robots_txt)
    app.router.add_get("/", ok_page)
    app.router.add_get("/page1", ok_page)
    app.router.add_get("/page2", ok_page)
    app.router.add_get("/private/secret", private_page)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()

    # достаём реальный порт
    sockets = list(site._server.sockets)  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    return runner, base


async def demo():
    runner, base = await _demo_server()
    try:
        async with AsyncCrawler(
            max_concurrent=5,
            requests_per_second=2.0,
            per_domain=True,
            respect_robots=True,
            min_delay=0.5,
            jitter=0.2,
            user_agent="MyBot/1.0",
        ) as crawler:
            urls = [
                f"{base}/",
                f"{base}/page1",
                f"{base}/page2",
                f"{base}/private/secret",  # должен блокироваться robots
            ]

            results = await asyncio.gather(*(crawler.fetch(u) for u in urls))
            for r in results:
                if r.blocked_by_robots:
                    logger.info(f"DEMO result | BLOCKED | url={r.url}")
                else:
                    logger.info(f"DEMO result | ok={r.ok} status={r.status} waited={r.waited:.2f}s url={r.url}")

            stats = await crawler.get_stats()
            logger.info(f"DEMO stats | {stats}")
    finally:
        await runner.cleanup()


# ---------------- main ----------------
if __name__ == "__main__":
    import sys
    import subprocess

    mode = (sys.argv[1] if len(sys.argv) > 1 else "demo").lower()

    if mode == "test":
        # гоняем отдельный файл тестов
        raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "-v", "tests.test_crawler_day4"]))
    else:
        asyncio.run(demo())