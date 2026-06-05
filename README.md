# PDF RAG — каркас

Только структура, синхронная база, миграция, Docker и контракты API.
PDF-парсинг, загрузка, CRUD, эмбеддинги, поиск и генерация **не реализованы**.
Сервисные функции поднимают `NotImplementedError`; все бизнес-маршруты
возвращают HTTP 501. Тестов, UI, датасетов и ML-зависимостей нет.

## Запуск

Нужен работающий Docker с Compose:

```sh
docker compose up --build
```

Пример `.env.example` необязателен: Compose содержит локальные демонстрационные
значения. При необходимости скопируйте его в `.env`. API-ключ не требуется.
API слушает только `127.0.0.1:8000`; PostgreSQL наружу не опубликован.
Демонстрационный пароль не предназначен для публичного развёртывания.

При старте API выполняется `alembic upgrade head`, затем `uvicorn`.
Одна миграция включает расширение `vector` и создаёт `documents`, `chunks`,
`query_logs`. Данные PostgreSQL сохраняются в именованном томе.

```sh
curl http://localhost:8000/health
# {"status":"ok"}
docker compose down
```

OpenAPI: http://localhost:8000/docs.

## Без Docker: Windows / PowerShell

Нужен локальный Python (целевой — 3.11). Из папки проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

В другом окне PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Это запуск **только API**: `/health` не подключается к PostgreSQL, а бизнес-маршруты
остаются заглушками. В отличие от Docker-команды, прямой запуск Uvicorn не запускает
миграцию автоматически. Остановка — `Ctrl+C` в окне сервера.

Для базы отдельно нужны PostgreSQL с установленным расширением pgvector, существующая
база и параметры доступа. До миграции задайте реальный DSN через `DATABASE_URL`
или локальный `.env`; пример ниже содержит только заменяемые значения:

```powershell
$env:DATABASE_URL='postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE'
.\.venv\Scripts\alembic upgrade head
```

Специальные символы пароля должны быть URL-кодированы. Не коммитьте `.env`.
Значение с хостом `db` из `.env.example` предназначено для Compose; для локального
сервера нужен `localhost`. Установка системной базы и расширения не выполняется
этими командами; пользователь базы должен иметь право создать расширение `vector`.

| Маршрут | Контракт |
| --- | --- |
| `GET /health` | `{"status":"ok"}` — проверка процесса API |
| `POST /documents` | multipart `file`: PDF, заглушка 501 |
| `GET /documents` | заглушка 501 |
| `DELETE /documents/{id}` | числовой ID, заглушка 501 |
| `GET /search?q=...&k=5` | `list[SearchHit]`, заглушка 501 |
| `POST /ask` | `{"question":"...","k":5}` → `AnswerResult`, заглушка 501 |
| `GET /history?limit=20` | заглушка 501 |

Настройки: `DATABASE_URL`, `EMBEDDING_MODEL=intfloat/multilingual-e5-small`,
`EMBEDDING_DIM=384`, `LLM_API_KEY`, `LLM_MODEL`, `CHUNK_SIZE=800`,
`CHUNK_OVERLAP=100`. Размерность колонки фиксирована: `Vector(384)`.
Префиксы будущих эмбеддингов: `query: ` и `passage: `.
Номера страниц начинаются с 1 и передаются в источники ответа.

## Статус проверки

После установки и запуска Docker пользователем выполнено
`docker compose up --build -d` на подключённом Windows-компьютере:

- Docker Engine 29.9.2, Compose 5.6.1, контекст `desktop-linux`.
- Образ API собран, оба контейнера работают; `db` имеет статус `healthy`.
- В API — Python 3.11.17; в базе — PostgreSQL 16.15 и pgvector 0.8.7.
- В реальной базе подтверждена версия Alembic `0001`, таблицы `documents`,
  `chunks`, `query_logs` и тип колонки `embedding`: `vector(384)`.
- `GET http://127.0.0.1:8000/health` вернул **HTTP 200** и `{"status":"ok"}`.

Критерий запуска каркаса подтверждён. Бизнес-функции остаются заглушками.
Unit/integration-тесты не написаны и не запускались согласно запросу.
Системные настройки, сторонние службы и нативная установка PostgreSQL не менялись.

Ранее выполненная проверка без Docker:

Для нативного запуска обнаружены Python 3.14 (3.11 не найден) и работающий
PostgreSQL 18 на `localhost:5432`, принимающий подключения по `pg_isready`.
PostgreSQL 16 не найден. Файлы `vector.control` и `vector.dll` в установке
PostgreSQL 18 отсутствуют; миграция и доступ к проектной базе не проверены.
Сервисы и системные настройки не менялись; SQLite не используется.

Нативное окружение `.venv` создано, зависимости из `requirements.txt` установлены.
Команда `uvicorn app.main:app --host 127.0.0.1 --port 8000` запустила API;
реальная HTTP-проверка `/health` вернула **200** и `{"status":"ok"}`.
Это подтверждает только запуск API на Python 3.14, а не работоспособность базы,
миграции или бизнес-маршрутов. Дополнительные тесты не запускались.

Предыдущая полная реализация целиком сохранена в локальной соседней резервной папке
вне Git-репозитория; в этот каркас она не включена.
