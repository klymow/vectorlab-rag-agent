# AI-оператор внутренней IT-поддержки «ВекторЛаб»

Учебное web-приложение реализует полный RAG-контур по русскоязычной базе знаний: локальные embeddings, независимые FAISS и BM25 поиски, объединение через Reciprocal Rank Fusion, проверяемый Final Top-K контекст и grounded-ответ Groq со ссылками на использованные источники.

Если найденный контекст не подтверждает ответ, приложение возвращает стандартное сообщение о недостаточности информации и не показывает вымышленные источники.

## Возможности

- автоматическая загрузка всех `*.md` из `knowledge_base/documents/`;
- извлечение `document_id`, заголовка, версии и исходного файла из существующего front matter;
- смысловой chunking по Markdown-заголовкам с детерминированными `chunk_id` и `content_hash`;
- локальная модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` без отправки документов во внешний embedding API;
- нормализованные `float32` embeddings и точный FAISS `IndexFlatIP`;
- отдельный BM25-поиск с токенизацией русского и технического текста;
- Vector Search и BM25 Search для каждого вопроса, deduplication по `chunk_id`, RRF и Final Top-K;
- передача в Groq только итогового контекста и проверка ссылок `[S1]`, `[S2]` в ответе;
- FastAPI backend и русскоязычный web UI с источниками и полной retrieval-диагностикой.

## Структура проекта

```text
app/
  config.py                   конфигурация без хранения секретов
  main.py                     FastAPI, API и статический интерфейс
  rag/                        загрузка, chunking, индексы, поиск, RRF и RAG
  static/                     HTML, CSS и JavaScript интерфейса
knowledge_base/documents/     12 исходных Markdown-документов
scripts/                      переиндексация и диагностические CLI-команды
storage/                      локальные генерируемые индексы, не входит в Git
tests/                        unit/API-тесты и контрольные вопросы
```

Исходные DOCX-файлы задания, `.env`, `.venv`, кэши и содержимое `storage/` исключены из Git.

## Запуск на Windows с нуля

Проверенная конфигурация: Python 3.12.10. Первая загрузка embedding-модели требует доступ в интернет; после сохранения модели Sentence Transformers работает локально. Для генерации ответа нужен Groq API key.

Из корня проекта выполните:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Откройте локальный `.env` и заполните только значение `GROQ_API_KEY`. Не добавляйте `.env` в Git. Значение `GROQ_MODEL=openai/gpt-oss-120b` уже задано в примере.

Постройте все локальные артефакты:

```powershell
.\.venv\Scripts\python.exe scripts\reindex.py
```

Ожидаемый результат для текущей базы: 12 документов, 62 чанка, embeddings размерности 384, FAISS `IndexFlatIP` и BM25 corpus из 62 чанков.

Запустите тесты и web-приложение:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Откройте `http://127.0.0.1:8000`. Endpoint `GET /api/health` проверяет backend без загрузки embedding-модели и без обращения к Groq. Первый `POST /api/ask` загружает локальную модель и может выполняться дольше последующих запросов.

Если порт 8000 занят, используйте, например:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8765
```

## Переиндексация базы знаний

Добавьте, измените или удалите Markdown-файл в `knowledge_base/documents/`, затем снова выполните:

```powershell
.\.venv\Scripts\python.exe scripts\reindex.py
```

Код менять не требуется. Команда заново создаёт `chunks.json`, embeddings, FAISS index и mapping, vector manifest, BM25 corpus и manifest в `storage/`. Эти артефакты намеренно не хранятся в Git.

Отдельные диагностические команды:

```powershell
.\.venv\Scripts\python.exe scripts\search_vector.py "Что делать при VPN-1207?" --top-n 5
.\.venv\Scripts\python.exe scripts\search_bm25.py "VPN-1207" --top-n 5
.\.venv\Scripts\python.exe scripts\search_hybrid.py "Что делать при VPN-1207?"
.\.venv\Scripts\python.exe scripts\ask.py "Что делать при VPN-1207?"
```

## Контрольные вопросы

Набор `tests/evaluation_questions.json` покрывает все 12 документов и содержит отдельные запросы, на которые база не должна отвечать. Полная LLM-проверка требует настроенного Groq и выполняется вручную через web UI или `scripts/ask.py`; unit/API-тесты не отправляют запросы во внешние сервисы.

Базовые smoke-сценарии:

- `Что делать при VPN-1207?` — достаточный ответ со ссылкой на `VL-IT-001`;
- `Меня слышат на совещании, но я не слышу коллег. Что проверить?` — раздел про устройство вывода в `VL-IT-008`;
- `Потерял рабочий ноутбук в командировке. Что делать?` — инструкции `VL-IT-010` и данные обращения `VL-IT-012`;
- `Что означает ошибка FS-403?` — достаточный ответ по `VL-IT-005`;
- `Как исправить CRM-999?` — insufficient/no-answer без источников.

## Соответствие обязательным требованиям PRO

| Требование | Реализация |
|---|---|
| Не менее 10 документов | 12 файлов в `knowledge_base/documents/` |
| Metadata документа и чанка | `app/rag/document_loader.py`, `app/rag/models.py`, `app/rag/chunker.py` |
| Автозагрузка и chunking | `load_documents()` и `chunk_documents()`; запуск из `scripts/reindex.py` |
| Локальные embeddings | `app/rag/local_embeddings.py` |
| Vector store и FAISS | `app/rag/vector_index.py`, `app/rag/vector_search.py`, `IndexFlatIP` |
| Отдельный BM25 | `app/rag/bm25_index.py`, `app/rag/bm25_search.py` |
| Автоматический поиск обоими каналами | `app/rag/hybrid_search.py` |
| RRF, deduplication и Final Top-K | `app/rag/fusion.py`, объединение по стабильному `chunk_id` |
| Контекст для LLM | `app/rag/context_builder.py`, `app/rag/service.py` |
| Ограничение базой и no-answer | `SYSTEM_PROMPT` в `app/rag/groq_client.py`, проверка ссылок в `app/rag/service.py` |
| Источники пользователю | `app/rag/service.py`, `app/static/app.js` |
| Все четыре вида диагностики | `app/main.py`, `app/static/index.html`, `app/static/app.js` |
| Переиндексация без изменения кода | `scripts/reindex.py` |
| Набор тестовых вопросов | `tests/evaluation_questions.json` |
| Web UI и end-to-end backend | `app/main.py`, `app/static/`; API-тесты в `tests/test_api.py` |

## Безопасность конфигурации

`.env.example` содержит только пустой placeholder `GROQ_API_KEY=`. Ключ загружается backend-компонентом из локального `.env`, передаётся только Groq SDK и не включается в API-ответы или frontend. Backend возвращает клиенту безопасный тип ошибки без текста внутреннего исключения.

