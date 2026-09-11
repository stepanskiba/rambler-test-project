# UGC Moderation Service

Сервис принимает текст комментария и возвращает решение модерации: `ALLOW`, `REVIEW` или `BLOCK`.

Модель: TF-IDF (символьные n-граммы) + логистическая регрессия,
обучена на [Mnwa/russian-toxic](https://huggingface.co/datasets/Mnwa/russian-toxic).
Разметка бинарная, поэтому без причин блокировки.

## Требования

Python 3.10.11, опционально docker.

## Установка

```bash
pip install -r requirements-dev.txt
```

Для запуска только сервиса достаточно `requirements.txt`.

## Обучение

Сейчас модель и датасеты есть в репозитории в гите, в настоящем сервисе стоит хранить в dvc.

Для обучения запустить:

```bash
python src/train.py
```

Создаёт `models/model.joblib`, `data/valid.jsonl` и `data/test.jsonl`. Занимает пару минут.


## Конфигурация

Читается из переменных окружения, для локального запуска - `cp .env.example .env`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MODEL_PATH` | `models/model.joblib` | путь к модели |
| `ALLOW_BELOW` | `0.33` | ниже порога - `ALLOW` |
| `BLOCK_ABOVE` | `0.88` | выше порога - `BLOCK` |

## Запуск

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8080
```

Docker (должна быть обученная модель):
```bash
docker build -t moderation .
docker run --env-file .env -p 8080:8080 moderation
```

## Выбор порогов

Подбор - в `notebooks/thresholds.ipynb`, на valid (20% от train)

Ограничения:

- ложно заблокированных не больше 0.5% от нормальных комментариев
- среди ALLOW токсичных не больше 1%
- в REVIEW не больше 10% потока

В valid и test токсичных ~21%, в реальных комментариях меньше - возьмем 5%.
От этой доли зависят доли зон, поэтому они пересчитаны под 5%.
Доля ложных блокировок среди нормальных от неё не зависит,
поэтому ограничение на BLOCK задано через неё.

Результат на valid:

```
ALLOW_BELOW=0.33  BLOCK_ABOVE=0.88

ALLOW  86.4%, токсичных в ней 0.38%
REVIEW 9.7%
BLOCK  3.9%, precision 88.2%, recall 68.8%, ложных блокировок 0.49%
```

`BLOCK_ABOVE` упирается в лимит ложных блокировок, `ALLOW_BELOW` - в бюджет REVIEW.
Лимит токсичных в ALLOW выполняется с запасом.

## Метрики

```bash
python src/eval.py [--model-path PATH] [--test-path PATH] [--metrics-path PATH]
```

Считает метрики по сохранённой модели на test, пишет результат в `data/metrics.json`.
Метрики привязаны к фактическим порогам сервиса:
доли зон, загрязнённость ALLOW, precision / recall / FPR для BLOCK, плюс ROC-AUC.

Результат на test (69 506 сообщений, пересчёт под 5% токсичных):

```
ROC-AUC 0.9756

ALLOW  86.4%, токсичных в ней 0.38%
REVIEW 9.6%
BLOCK  3.9%, precision 87.7%, recall 69.1%, ложных блокировок 0.51%
```

Ложных блокировок на test 0.51% при лимите 0.5%.

## Тесты

```bash
pytest
```

Проверяют контракт API, валидацию входа и границы зон.
Требуют обученной модели.

## Нагрузочное тестирование

Сервис должен быть запущен:

```bash
python tests/loadtest.py [--url http://localhost:8080] [--n 5000] [--concurrency 16]
```

Отправляет через `POST /moderate` случайные `--n` сообщений из `data/test.jsonl`
(сид фиксирован, выборка одна и та же), `--n 0` - весь test.
Результат пишется в `data/loadtest.json`: throughput в messages/min (по успешным ответам),
латентность p50 / p95 / p99, ошибки и доли решений.

Результат (1 воркер uvicorn, concurrency 16, 5000 сообщений):

```
throughput: ~6 500 messages/min
latency p50 / p95 / p99: 61 / 595 / 1121 ms
errors: 0

ALLOW 72.2%, REVIEW 11.9%, BLOCK 15.9%
```

Хвост латентности - в основном ожидание в очереди: 16 одновременных запросов
на один процесс, предсказание упирается в одно ядро CPU.

Доли решений отражают долю токсичных в test (~21%). Под ожидаемые 5% в потоке
их пересчитывает `src/eval.py` - там REVIEW ~9.6%.

## API

```bash
curl -X POST localhost:8080/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text": "привет, отличная статья"}'
```

```json
{"decision": "ALLOW", "confidence": 0.03}
```

- `decision` - `ALLOW` / `REVIEW` / `BLOCK`
- `confidence` - скор токсичности
- текст: 1–5000 символов, иначе `422`

`GET /health` - стандартный хелсчек. Документация: `/docs`.
