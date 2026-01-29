# html_parser.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger("html_parser")


def is_valid_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


@dataclass
class ParsedImage:
    src: str
    alt: str = ""


class HTMLParser:
    """
    Парсит HTML и извлекает структуру:
    - title/metadata
    - text (грубая "основная" выжимка)
    - links (абсолютные URL)
    - images (src/alt)
    - headings (h1/h2/h3)
    - tables (строки/ячейки как текст)
    - lists (ul/ol как списки строк)
    """

    async def parse_html(self, html: str, url: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "metadata": {},
            "images": [],
            "headings": {"h1": [], "h2": [], "h3": []},
            "tables": [],
            "lists": {"ul": [], "ol": []},
            "warnings": [],
        }

        try:
            soup = BeautifulSoup(html or "", "lxml")
        except Exception as e:
            # критическая ошибка парсинга — возвращаем частичный результат
            msg = f"BeautifulSoup parse error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)
            return result

        # Метаданные и title не должны валить весь парсинг
        try:
            result["metadata"] = self.extract_metadata(soup)
            result["title"] = result["metadata"].get("title", "") or ""
        except Exception as e:
            msg = f"metadata extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        # Текст
        try:
            result["text"] = self.extract_text(soup)
        except Exception as e:
            msg = f"text extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        # Ссылки
        try:
            result["links"] = self.extract_links(soup, base_url=url)
        except Exception as e:
            msg = f"links extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        # Специфичные данные
        try:
            result["images"] = self.extract_images(soup, base_url=url)
        except Exception as e:
            msg = f"images extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        try:
            result["headings"] = self.extract_headings(soup)
        except Exception as e:
            msg = f"headings extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        try:
            result["tables"] = self.extract_tables(soup)
        except Exception as e:
            msg = f"tables extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        try:
            result["lists"] = self.extract_lists(soup)
        except Exception as e:
            msg = f"lists extraction error: {e}"
            logger.warning(msg)
            result["warnings"].append(msg)

        return result

    def extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
        *,
        same_domain_only: bool = False,
    ) -> List[str]:
        links: List[str] = []
        seen = set()

        base_parsed = urlparse(base_url)
        base_host = base_parsed.netloc.lower()

        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue

            # фильтруем мусорные схемы
            lowered = href.lower()
            if lowered.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue

            abs_url = urljoin(base_url, href)

            # иногда urljoin оставляет странные штуки — проверим
            if not is_valid_http_url(abs_url):
                continue

            if same_domain_only:
                host = urlparse(abs_url).netloc.lower()
                if host != base_host:
                    continue

            if abs_url not in seen:
                seen.add(abs_url)
                links.append(abs_url)

        return links

    def extract_text(self, soup: BeautifulSoup, selector: str | None = None) -> str:
        # выкидываем заведомо “не текст”
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        root = soup
        if selector:
            found = soup.select_one(selector)
            if found is not None:
                root = found

        text = root.get_text(separator="\n", strip=True)
        # чуть подчистим пустые строки
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        meta: Dict[str, str] = {}

        # title
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            meta["title"] = title_tag.get_text(strip=True)

        # description / keywords / og:*
        for m in soup.find_all("meta"):
            name = (m.get("name") or "").strip().lower()
            prop = (m.get("property") or "").strip().lower()
            content = (m.get("content") or "").strip()

            key = name or prop
            if not key or not content:
                continue

            # нас интересует базовый набор + ог-ки
            if key in ("description", "keywords") or key.startswith("og:"):
                meta[key] = content

        return meta

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        images: List[Dict[str, str]] = []
        seen = set()

        for img in soup.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue
            abs_src = urljoin(base_url, src)
            if not is_valid_http_url(abs_src):
                continue
            if abs_src in seen:
                continue
            seen.add(abs_src)

            alt = (img.get("alt") or "").strip()
            images.append({"src": abs_src, "alt": alt})

        return images

    def extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        res = {"h1": [], "h2": [], "h3": []}
        for level in ("h1", "h2", "h3"):
            for h in soup.find_all(level):
                txt = h.get_text(" ", strip=True)
                if txt:
                    res[level].append(txt)
        return res

    def extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """
        Возвращает список таблиц.
        Каждая таблица: список строк.
        Каждая строка: список ячеек (текст).
        """
        tables: List[List[List[str]]] = []

        for table in soup.find_all("table"):
            parsed_rows: List[List[str]] = []
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                row = [c.get_text(" ", strip=True) for c in cells]
                if any(cell for cell in row):
                    parsed_rows.append(row)
            if parsed_rows:
                tables.append(parsed_rows)

        return tables

    def extract_lists(self, soup: BeautifulSoup) -> Dict[str, List[List[str]]]:
        """
        Возвращает:
        - ul: список списков (каждый ul -> список li)
        - ol: аналогично
        """
        out: Dict[str, List[List[str]]] = {"ul": [], "ol": []}

        for tagname in ("ul", "ol"):
            for lst in soup.find_all(tagname):
                items = []
                for li in lst.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        items.append(txt)
                if items:
                    out[tagname].append(items)

        return out