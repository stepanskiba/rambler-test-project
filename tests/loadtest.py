import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
import pandas as pd

import httpx

MAX_LEN = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--data-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--out-path", type=Path, default=Path("data/loadtest.json"))
    parser.add_argument("--n", type=int, default=5000, help="сколько сообщений отправить, 0 - весь набор")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()


def load_texts(path: Path, n: int) -> list[str]:
    texts = pd.read_json(path, lines=True)["text"].fillna("").str.strip().str[:MAX_LEN]
    texts = texts[texts != ""]
    if n:
        texts = texts.sample(n=min(n, len(texts)), random_state=42)
    return texts.tolist()

def percentile(sorted_values: list[float], q: float) -> float:
    idx = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


async def run(args: argparse.Namespace, texts: list[str]) -> dict:
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=30) as client:
        for text in texts[: args.warmup]:
            await client.post("/moderate", json={"text": text})

        queue: asyncio.Queue[str] = asyncio.Queue()
        for text in texts:
            queue.put_nowait(text)

        latencies: list[float] = []
        decisions: Counter[str] = Counter()
        errors: Counter[str] = Counter()

        async def worker() -> None:
            while not queue.empty():
                text = queue.get_nowait()
                start = time.perf_counter()
                try:
                    r = await client.post("/moderate", json={"text": text})
                except httpx.HTTPError as e:
                    errors[type(e).__name__] += 1
                    continue
                latencies.append(time.perf_counter() - start)
                if r.status_code == 200:
                    decisions[r.json()["decision"]] += 1
                else:
                    errors[str(r.status_code)] += 1

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - start

    ok = sum(decisions.values())
    lat_ms = sorted(x * 1000 for x in latencies)
    return {
        "url": args.url,
        "messages": len(texts),
        "concurrency": args.concurrency,
        "elapsed_sec": round(elapsed, 2),
        "throughput_msg_per_min": round(ok / elapsed * 60),
        "latency_ms": {
            "p50": round(percentile(lat_ms, 0.50), 1),
            "p95": round(percentile(lat_ms, 0.95), 1),
            "p99": round(percentile(lat_ms, 0.99), 1),
        } if lat_ms else {},
        "ok": ok,
        "errors": dict(errors),
        "decisions": {k: round(v / ok, 4) for k, v in sorted(decisions.items())} if ok else {},
    }


def main() -> None:
    args = parse_args()
    texts = load_texts(args.data_path, args.n)
    result = asyncio.run(run(args, texts))

    args.out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()