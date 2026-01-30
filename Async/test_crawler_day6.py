import os
import json
import csv
import io
import tempfile
import unittest

import aiofiles
import aiosqlite

from crawler_day6 import JSONStorage, CSVStorage, SQLiteStorage, save_with_retry


EXPECTED_KEYS = {
    "url", "title", "text", "links", "metadata", "crawled_at", "status_code", "content_type"
}


def assert_schema(testcase: unittest.TestCase, obj: dict) -> None:
    testcase.assertTrue(EXPECTED_KEYS.issubset(set(obj.keys())), f"missing keys: {EXPECTED_KEYS - set(obj.keys())}")
    testcase.assertIsInstance(obj["url"], str)
    testcase.assertIsInstance(obj["title"], str)
    testcase.assertIsInstance(obj["text"], str)
    testcase.assertIsInstance(obj["links"], list)
    testcase.assertIsInstance(obj["metadata"], dict)
    testcase.assertIsInstance(obj["crawled_at"], str)
    testcase.assertIsInstance(obj["status_code"], int)
    testcase.assertIsInstance(obj["content_type"], str)


class TestJSONStorage(unittest.IsolatedAsyncioTestCase):
    async def test_json_ndjson_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.ndjson")
            st = JSONStorage(path, flush_every=1)

            await st.save({"url": "u1", "text": "t1", "status_code": 200, "links": ["a"], "metadata": {"k": 1}})
            await st.save({"url": "u2", "text": "t2", "status_code": 500, "links": [], "metadata": {"k": 2}})
            await st.close()

            lines = []
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                async for line in f:
                    if line.strip():
                        lines.append(line)

            self.assertEqual(len(lines), 2)

            obj0 = json.loads(lines[0])
            obj1 = json.loads(lines[1])

            assert_schema(self, obj0)
            assert_schema(self, obj1)


class TestCSVStorage(unittest.IsolatedAsyncioTestCase):
    async def test_csv_read_back_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.csv")
            st = CSVStorage(path, flush_every=1)

            await st.save({"url": "u1", "text": "hello, world", "links": ["a", "b"], "metadata": {"k": 1}})
            await st.save({"url": "u2", "text": "line\nbreak", "links": [], "metadata": {"k": 2}})
            await st.close()

            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()

            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            self.assertEqual(len(rows), 2)

            for r in rows:
                r["links"] = json.loads(r.get("links") or "[]")
                r["metadata"] = json.loads(r.get("metadata") or "{}")
                r["status_code"] = int(r.get("status_code") or 0)
                # title/content_type могут быть пустыми -> это ок
                assert_schema(self, r)


class TestSQLiteStorage(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_insert_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.sqlite")
            st = SQLiteStorage(path, batch_size=2, flush_interval=0.1)

            await st.save({"url": "u1", "text": "t1", "links": ["a"], "metadata": {"k": 1}})
            await st.save({"url": "u2", "text": "t2", "links": [], "metadata": {"k": 2}})
            await st.save({"url": "u3", "text": "t3", "links": ["x", "y"], "metadata": {"k": 3}})
            await st.close()

            async with aiosqlite.connect(path) as db:
                async with db.execute("SELECT COUNT(*) FROM pages;") as cur:
                    row = await cur.fetchone()
                self.assertEqual(row[0], 3)

                async with db.execute("""
                    SELECT url, title, text, links_json, metadata_json, crawled_at, status_code, content_type
                    FROM pages ORDER BY id LIMIT 1;
                """) as cur:
                    one = await cur.fetchone()

            url, title, text, links_json, metadata_json, crawled_at, status_code, content_type = one
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
            assert_schema(self, obj)


class FlakyStorage:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.saved = 0

    async def save(self, data):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("temp fail")
        self.saved += 1

    async def close(self):
        return


class TestSaveRetry(unittest.IsolatedAsyncioTestCase):
    async def test_save_with_retry_eventually_ok(self):
        st = FlakyStorage(fail_times=2)
        await save_with_retry(st, {"url": "u1"}, max_retries=5, base_delay=0.01, backoff_factor=1.2)
        self.assertEqual(st.saved, 1)

    async def test_save_with_retry_exhausts(self):
        st = FlakyStorage(fail_times=10)
        with self.assertRaises(OSError):
            await save_with_retry(st, {"url": "u1"}, max_retries=2, base_delay=0.01, backoff_factor=1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)