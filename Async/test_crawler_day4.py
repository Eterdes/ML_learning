# tests/test_crawler_day4.py
import asyncio
import time
import unittest

import aiohttp
from aiohttp import web

from crawler_day4 import RateLimiter, RobotsParser, AsyncCrawler


async def _demo_server():
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

    sockets = list(site._server.sockets)  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    return runner, base


class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limiter_single_domain(self):
        rl = RateLimiter(requests_per_second=5.0, per_domain=True)  # interval 0.2s
        d = "example.com"
        t0 = time.monotonic()
        await rl.acquire(d)
        await rl.acquire(d)
        await rl.acquire(d)
        dt = time.monotonic() - t0
        # минимум ~0.4s (2 интервала ожидания между 3 запросами)
        self.assertGreaterEqual(dt, 0.35)

    async def test_rate_limiter_multi_domain_independent(self):
        rl = RateLimiter(requests_per_second=2.0, per_domain=True)  # 0.5s per domain
        t0 = time.monotonic()
        await rl.acquire("a.com")
        await rl.acquire("b.com")
        dt = time.monotonic() - t0
        # разные домены => почти без ожидания
        self.assertLess(dt, 0.2)

    async def test_rate_limiter_global(self):
        rl = RateLimiter(requests_per_second=4.0, per_domain=False)  # interval 0.25s global
        t0 = time.monotonic()
        await rl.acquire("a.com")
        await rl.acquire("b.com")
        await rl.acquire("c.com")
        dt = time.monotonic() - t0
        self.assertGreaterEqual(dt, 0.45)


class TestRobotsAndCrawler(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner, self.base = await _demo_server()

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_robots_parsing_and_block(self):
        async with aiohttp.ClientSession() as s:
            rp = RobotsParser(s)
            await rp.fetch_robots(self.base)
            self.assertTrue(rp.can_fetch(f"{self.base}/page1", user_agent="MyBot/1.0"))
            self.assertFalse(rp.can_fetch(f"{self.base}/private/secret", user_agent="MyBot/1.0"))
            await rp.fetch_robots(self.base)
            d = rp.get_crawl_delay(user_agent="MyBot/1.0")
            self.assertGreaterEqual(d, 1.0)

    async def test_crawler_blocks_disallowed_url(self):
        async with AsyncCrawler(
            max_concurrent=3,
            requests_per_second=10.0,
            respect_robots=True,
            min_delay=0.0,
            user_agent="MyBot/1.0",
        ) as c:
            r = await c.fetch(f"{self.base}/private/secret")
            self.assertTrue(r.blocked_by_robots)
            stats = await c.get_stats()
            self.assertEqual(stats["robots_blocked"], 1)

    async def test_crawler_respects_crawl_delay(self):
        async with AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,  # убираем влияние rate limiter
            respect_robots=True,        # crawl-delay=1
            min_delay=0.0,
            jitter=0.0,
            user_agent="MyBot/1.0",
        ) as c:
            t0 = time.monotonic()
            r1 = await c.fetch(f"{self.base}/page1")
            r2 = await c.fetch(f"{self.base}/page2")
            dt = time.monotonic() - t0
            self.assertTrue(r1.ok and r2.ok)
            # 2 запроса * 1s (мы ждём перед каждым запросом)
            self.assertGreaterEqual(dt, 2.0 - 0.2)


if __name__ == "__main__":
    unittest.main()