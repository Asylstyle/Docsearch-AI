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

На подключённом Windows-компьютере Docker не найден в PATH, стандартных каталогах
Docker Desktop, Chocolatey/Scoop и списке служб. Docker не устанавливался,
параметры системы не менялись. Поэтому `docker compose up --build`, миграция
на реальном PostgreSQL и HTTP-проверка `/health` **не выполнялись**.
Критерий готовности по запуску пока не подтверждён. Unit/integration-тесты
не написаны и не запускались согласно запросу.

Предыдущая полная реализация целиком сохранена в локальной соседней резервной папке
вне Git-репозитория; в этот каркас она не включена.
