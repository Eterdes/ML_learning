import unittest
import asyncio

import aiohttp

from crawler_day5 import RetryStrategy, AsyncCrawler, TransientError, PermanentError


class TestRetryStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_retry_on_transient(self):
        calls = 0

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TransientError()
            return "ok"

        retry = RetryStrategy(max_retries=3, backoff_factor=2.0, base_delay=0.01)
        res, attempts = await retry.execute_with_retry(flaky)

        self.assertEqual(res, "ok")
        self.assertEqual(attempts, 3)

    async def test_no_retry_on_permanent(self):
        async def bad():
            raise PermanentError()

        retry = RetryStrategy(max_retries=3)

        with self.assertRaises(PermanentError):
            await retry.execute_with_retry(bad)

    async def test_backoff_grows(self):
        retry = RetryStrategy(base_delay=0.1, backoff_factor=2.0)
        d1 = retry._delay(1, TransientError())
        d2 = retry._delay(2, TransientError())
        d3 = retry._delay(3, TransientError())
        self.assertGreater(d2, d1)
        self.assertGreater(d3, d2)


class TestCrawlerHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_404_no_retry(self):
        retry = RetryStrategy(max_retries=3, base_delay=0.01)
        crawler = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=3))

        r = await crawler.fetch("https://httpbin.org/status/404")
        self.assertFalse(r.ok)
        self.assertEqual(r.error_type, "PermanentError")

    async def test_503_retry_happens(self):
        retry = RetryStrategy(max_retries=2, base_delay=0.01)
        crawler = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=3))

        r = await crawler.fetch("https://httpbin.org/status/503")
        self.assertFalse(r.ok)
        # хотя бы 1 transient error зафиксирован
        self.assertGreaterEqual(retry.stats.errors_by_type.get("TransientError", 0), 1)

class TestTimeoutAndStats(unittest.IsolatedAsyncioTestCase):
    async def test_retry_on_timeout(self):
        """
        Проверяем, что asyncio.TimeoutError классифицируется как TransientError
        и что RetryStrategy делает повторы.
        """
        calls = 0

        async def always_timeout():
            nonlocal calls
            calls += 1
            raise asyncio.TimeoutError()

        retry = RetryStrategy(max_retries=2, base_delay=0.01)

        with self.assertRaises(TransientError):
            # оборачиваем таймаут как это делает crawler.fetch_url
            async def wrapper():
                try:
                    return await always_timeout()
                except asyncio.TimeoutError as e:
                    raise TransientError() from e

            await retry.execute_with_retry(wrapper)

        # 1 попытка + 2 повтора = 3 вызова
        self.assertEqual(calls, 3)

    async def test_error_stats_and_permanent_urls(self):
        """
        Проверяем, что:
        - статистика ошибок ведётся
        - permanent_error_urls пополняется
        """
        retry = RetryStrategy(max_retries=1, base_delay=0.01)
        crawler = AsyncCrawler(retry_strategy=retry, timeout=aiohttp.ClientTimeout(total=3))

        r1 = await crawler.fetch("https://httpbin.org/status/404")
        self.assertFalse(r1.ok)
        self.assertEqual(r1.error_type, "PermanentError")

        r2 = await crawler.fetch("https://httpbin.org/status/503")
        self.assertFalse(r2.ok)
        self.assertEqual(r2.error_type, "TransientError")

        # проверяем счетчики
        self.assertGreaterEqual(retry.stats.errors_by_type.get("PermanentError", 0), 1)
        self.assertGreaterEqual(retry.stats.errors_by_type.get("TransientError", 0), 1)

        # проверяем список permanent URL
        self.assertIn("https://httpbin.org/status/404", retry.stats.permanent_error_urls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
