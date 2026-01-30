# Advanced Async Web Crawler (Day 7)

Асинхронный веб-краулер на Python с поддержкой конкурентного обхода сайтов, sitemap.xml,
гибкой конфигурации, расширенной статистики и тестирования производительности.

Проект выполнен в учебных целях и демонстрирует архитектуру production‑подобного
асинхронного краулера.

---

## Возможности

- Асинхронная загрузка страниц (`asyncio`, `aiohttp`)
- Ограничения:
  - максимальное количество страниц (`max_pages`)
  - глубина обхода (`max_depth`)
  - конкурентность (`max_concurrent`)
- Поддержка `sitemap.xml` и `sitemapindex`
- Retry‑логика с экспоненциальным backoff
- Rate limit запросов
- Фильтрация URL (allowed / include / exclude)
- Хранение результатов:
  - NDJSON
  - CSV
  - SQLite
- Экспорт статистики:
  - JSON
  - HTML‑отчёт
- CLI‑интерфейс
- Мониторинг прогресса в реальном времени
- Тестирование производительности и функциональные тесты

---

## Архитектура

Основные компоненты проекта:

- **AdvancedCrawler** — основной orchestrator
- **AsyncCrawlerEngine** — HTTP‑клиент, retry и сохранение результатов
- **RetryStrategy** — политика повторных попыток
- **SitemapParser** — загрузка и разбор sitemap
- **CrawlerStats** — сбор и агрегация статистики
- **DataStorage** — абстракция хранилища данных
- **RateLimiter** — ограничение скорости запросов

---

## Установка

Требования:
- Python **3.10+**

Установка зависимостей:
```bash
pip install aiohttp aiofiles aiosqlite pyyaml
```

---

## Быстрый старт (CLI)

```bash
python crawler_day7.py \
  --urls https://example.com \
  --max-pages 100 \
  --max-depth 2 \
  --output results.ndjson
```

### Использование sitemap

```bash
python crawler_day7.py \
  --sitemap https://example.com/sitemap.xml \
  --max-pages 500
```

### Запуск с конфигурационным файлом

```bash
python crawler_day7.py --config config.yaml
```

---

## Использование в коде

```python
from crawler_day7 import AdvancedCrawler

async def main():
    crawler = AdvancedCrawler.from_config("config.yaml")
    await crawler.crawl()

    stats = crawler.get_stats()
    print(stats)

    crawler.export_to_html_report("report.html")
    await crawler.close()
```

---

## Конфигурация

Поддерживаются форматы **YAML** и **JSON**.

### Пример `config.yaml`

```yaml
start_urls:
  - https://example.com

sitemap_urls:
  - https://example.com/sitemap.xml

max_pages: 500
max_depth: 2
max_concurrent: 20
rate_limit: 0

timeout_total: 15
user_agent: crawler_day7/1.0

allowed_domains:
  - example.com

exclude_patterns:
  - logout
  - signup

storage: ndjson
output: results.ndjson

stats_json: stats.json
report_html: report.html

log_file: crawler.log
log_level: INFO

monitor_interval: 1.0
```

### Основные параметры

| Параметр | Описание |
|--------|--------|
| `start_urls` | Начальные URL |
| `sitemap_urls` | Sitemap URL |
| `max_pages` | Лимит страниц |
| `max_depth` | Максимальная глубина |
| `max_concurrent` | Количество воркеров |
| `rate_limit` | Запросов в секунду |
| `storage` | ndjson / csv / sqlite / none |
| `monitor_interval` | Интервал логирования прогресса |

---

## CLI параметры

```text
--urls               стартовые URL
--sitemap            sitemap URL
--max-pages          лимит страниц
--max-depth          глубина обхода
--output             файл результата
--config             файл конфигурации
--rate-limit         лимит запросов
--log-level          уровень логирования
--storage            тип хранилища
```

CLI‑параметры имеют приоритет над конфигурационным файлом.

---

## Статистика и отчёты

Собираемая статистика:
- обработанные страницы
- успешные и неуспешные запросы
- распределение HTTP‑статусов
- средняя скорость (pages/sec)
- время работы
- топ доменов

### HTML‑отчёт
Генерируется HTML‑отчёт с таблицами и графиками.

---

## Тестирование

### Производительность

Проведены тесты:
- синхронный краулер (`urllib`)
- асинхронный краулер (`aiohttp`)
- объёмы: **100 / 500 / 1000 страниц**

Измерено:
- скорость (pages/sec)
- пиковое потребление памяти (`tracemalloc`)

Async‑подход показывает ускорение ~2–3× по сравнению с sync.

### Функциональные тесты

Реализованы тесты:
- строгий контроль `max_pages`
- стабильность мониторинга
- рекурсивный разбор sitemap index

Запуск:
```bash
python -m unittest -v test_crawler_day7.py
```

---

## Итог

Проект демонстрирует:
- корректную асинхронную архитектуру
- масштабируемость
- стабильную работу на больших объёмах
- тестируемость и наблюдаемость