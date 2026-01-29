import unittest

from crawler_day3 import AsyncCrawler, CrawlerQueue, FetchResult


class TestCrawlerDay3(unittest.IsolatedAsyncioTestCase):
    async def test_queue_priority(self):
        q = CrawlerQueue()
        q.add_url("https://a.com/low", priority=0)
        q.add_url("https://a.com/high", priority=10)

        first = await q.get_next()
        self.assertEqual(first, "https://a.com/high")

    async def test_depth_limit(self):
        pages = {
            "https://site.com/": '<a href="/a">a</a>',
            "https://site.com/a": '<a href="/b">b</a>',
            "https://site.com/b": "end",
        }

        async def fake_fetch(_session, url: str):
            html = pages.get(url)
            if html is None:
                return FetchResult(False, 404, None, "404", final_url=url, content_type="text/html")
            return FetchResult(True, 200, html, None, final_url=url, content_type="text/html")

        crawler = AsyncCrawler(
            max_concurrent=3,
            max_depth=1,              # <= ключевое
            same_domain_only=True,
            fetch_override=fake_fetch,
        )

        res = await crawler.crawl(start_urls=["https://site.com/"], max_pages=10)

        # depth=0: /
        # depth=1: /a
        # depth=2: /b  -> НЕ должен попасть
        self.assertIn("https://site.com/", res)
        self.assertIn("https://site.com/a", res)
        self.assertNotIn("https://site.com/b", res)

    async def test_url_filtering(self):
        pages = {
            "https://site.com/": '<a href="/keep">k</a><a href="/drop.pdf">p</a><a href="/x">x</a>',
            "https://site.com/keep": "ok",
            "https://site.com/x": "ok",
        }

        async def fake_fetch(_session, url: str):
            html = pages.get(url)
            if html is None:
                return FetchResult(False, 404, None, "404", final_url=url, content_type="text/html")
            return FetchResult(True, 200, html, None, final_url=url, content_type="text/html")

        crawler = AsyncCrawler(
            max_concurrent=3,
            max_depth=2,
            same_domain_only=True,
            exclude_patterns=[r"\.pdf$"],      # выкинуть pdf
            include_patterns=[r"^https://site\.com/$", r"/keep$"],      
            fetch_override=fake_fetch,
        )

        res = await crawler.crawl(start_urls=["https://site.com/"], max_pages=10)

        self.assertIn("https://site.com/keep", res)
        self.assertNotIn("https://site.com/x", res)
        self.assertNotIn("https://site.com/drop.pdf", res)

    async def test_no_duplicates_in_visited(self):
        pages = {
            "https://site.com/": '<a href="/a">a</a><a href="/a">a2</a><a href="/a#frag">a3</a>',
            "https://site.com/a": "ok",
        }

        async def fake_fetch(_session, url: str):
            html = pages.get(url, "")
            return FetchResult(True, 200, html, None, final_url=url, content_type="text/html")

        crawler = AsyncCrawler(
            max_concurrent=3,
            max_depth=2,
            same_domain_only=True,
            fetch_override=fake_fetch,
        )

        res = await crawler.crawl(start_urls=["https://site.com/"], max_pages=10)

        # visited_urls — set => дублей быть не может
        self.assertIn("https://site.com/a", crawler.visited_urls)

        # и по факту страница /a должна быть обработана один раз
        self.assertIn("https://site.com/a", res)
        self.assertEqual(1, sum(1 for u in res.keys() if u == "https://site.com/a"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
