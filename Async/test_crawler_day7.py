import unittest
import asyncio
from typing import Tuple
from aiohttp import web
import aiohttp

from crawler_day7 import AdvancedCrawler, CrawlerConfig, RetryStrategy, AsyncCrawlerEngine, SitemapParser
from aiohttp.web import AppKey

def make_site_app(total_pages: int, fanout: int = 10) -> web.Application:
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        path = request.path
        if path == "/":
            i = 0
        else:
            try:
                i = int(path.split("/")[-1])
            except Exception:
                return web.Response(status=404, text="not found")

        if i < 0 or i >= total_pages:
            return web.Response(status=404, text="not found")

        links = []
        for k in range(1, fanout + 1):
            j = i + k
            if j < total_pages:
                links.append(f'<a href="/p/{j}">p{j}</a>')

        html = f"<html><body><h1>page {i}</h1>{' '.join(links)}</body></html>"
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", handler)
    app.router.add_get("/p/{i}", handler)
    return app


async def start_server(app: web.Application) -> Tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def stop_server(runner: web.AppRunner) -> None:
    await runner.cleanup()


class TestCrawlerDay7Functional(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # quiet logger
        import logging
        self.logger = logging.getLogger("test")
        self.logger.handlers.clear()
        self.logger.propagate = False
        self.logger.setLevel(logging.CRITICAL)

    async def _make_crawler(self, base_url: str, *, max_pages: int, max_depth: int, concurrent: int = 20) -> AdvancedCrawler:
        cfg = CrawlerConfig(
            start_urls=[base_url + "/"],
            sitemap_urls=[],
            max_pages=max_pages,
            max_depth=max_depth,
            max_concurrent=concurrent,
            rate_limit=0.0,
            timeout_total=10.0,
            respect_robots=False,
            storage="none",
            output="ignore.ndjson",
            stats_json="ignore_stats.json",
            report_html="ignore_report.html",
            log_file="ignore.log",
            log_level="ERROR",
            # important for monitor
            monitor_interval=0.2,
        )

        retry = RetryStrategy(max_retries=1, backoff_factor=1.0, base_delay=0.01)
        engine = AsyncCrawlerEngine(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=cfg.timeout_total), storage=None)
        return AdvancedCrawler(cfg, engine, self.logger)

    async def test_max_pages_strict(self):
        # Create many pages but max_pages small; processed must be exactly max_pages.
        app = make_site_app(total_pages=200, fanout=10)
        runner, base = await start_server(app)
        try:
            crawler = await self._make_crawler(base, max_pages=30, max_depth=200, concurrent=50)
            try:
                await crawler.crawl()
                stats = crawler.get_stats()
                self.assertEqual(stats["processed_pages"], 30)
                self.assertEqual(stats["successful"] + stats["failed"], 30)
            finally:
                await crawler.close()
        finally:
            await stop_server(runner)

    async def test_monitor_does_not_crash(self):
        # If monitor loop references missing cfg fields, crawl would crash.
        app = make_site_app(total_pages=50, fanout=5)
        runner, base = await start_server(app)
        try:
            crawler = await self._make_crawler(base, max_pages=20, max_depth=50, concurrent=10)
            try:
                await crawler.crawl()
                stats = crawler.get_stats()
                self.assertEqual(stats["processed_pages"], 20)
            finally:
                await crawler.close()
        finally:
            await stop_server(runner)

    async def test_sitemap_index_recursive(self):
        base_holder = {"base": ""}   # будет заполнено после старта

        app = web.Application()

        async def sitemap_index(request):
            b = base_holder["base"]
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap><loc>{b}/sitemap1.xml</loc></sitemap>
    <sitemap><loc>{b}/sitemap2.xml</loc></sitemap>
    </sitemapindex>
    """
            return web.Response(text=xml, content_type="application/xml")

        async def sitemap1(request):
            b = base_holder["base"]
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>{b}/a</loc></url>
    <url><loc>{b}/b</loc></url>
    </urlset>
    """
            return web.Response(text=xml, content_type="application/xml")

        async def sitemap2(request):
            b = base_holder["base"]
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>{b}/c</loc></url>
    </urlset>
    """
            return web.Response(text=xml, content_type="application/xml")

        async def page(request):
            return web.Response(text="<html><body>ok</body></html>", content_type="text/html")

        app.router.add_get("/sitemap.xml", sitemap_index)
        app.router.add_get("/sitemap1.xml", sitemap1)
        app.router.add_get("/sitemap2.xml", sitemap2)
        app.router.add_get("/a", page)
        app.router.add_get("/b", page)
        app.router.add_get("/c", page)

        runner, base = await start_server(app)
        base_holder["base"] = base  # <-- не app[...], warning не будет
        
        try:
            # Minimal engine for sitemap fetch
            cfg = CrawlerConfig(
                start_urls=[],
                sitemap_urls=[],
                max_pages=10,
                max_depth=1,
                max_concurrent=5,
                rate_limit=0.0,
                timeout_total=10.0,
                respect_robots=False,
                storage="none",
                output="ignore.ndjson",
                stats_json="ignore_stats.json",
                report_html="ignore_report.html",
                log_file="ignore.log",
                log_level="ERROR",
                monitor_interval=0.2,
            )
            retry = RetryStrategy(max_retries=1, backoff_factor=1.0, base_delay=0.01)
            engine = AsyncCrawlerEngine(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=cfg.timeout_total), storage=None)
            smp = SitemapParser(engine, self.logger)

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cfg.timeout_total)) as session:
                urls = await smp.fetch_sitemap(session, base + "/sitemap.xml")

            self.assertEqual(set(urls), {base + "/a", base + "/b", base + "/c"})
        finally:
            await stop_server(runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
