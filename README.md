# UGC Moderation Service

Сервис принимает текст комментария и возвращает решение: `ALLOW`, `REVIEW` или `BLOCK`.

Модель: TF-IDF (символьные n-граммы) + логистическая регрессия,
обучена на [Mnwa/russian-toxic](https://huggingface.co/datasets/Mnwa/russian-toxic).

## Обучение

```bash
pip install -r requirements.txt
python src/train.py
```

## Запуск

Локально:
```bash
uvicorn src.app:app --port 8080
```

Docker:
```bash
docker build -t moderation .
docker run -p 8080:8080 moderation
```

## API

```bash
curl -X POST localhost:8080/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text": "привет, отличная статья"}'
```

Ответ:
```json
{"decision": "ALLOW", "confidence": 0.03}
```

- `decision` — `ALLOW` / `REVIEW` / `BLOCK`
- `confidence` — уверенность в решении, 0..1