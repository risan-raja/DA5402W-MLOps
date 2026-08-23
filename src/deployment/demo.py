"""POST demo wavs to the live API so Prometheus/Grafana record traffic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

from src.models.runtime_env import ROOT

SAMPLE_DIR = ROOT / "data" / "sample_audio"
CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)


def iter_demo_wavs(sample_dir: Path | None = None) -> list[tuple[str, Path]]:
    """Return ``(expected_class, wav_path)`` for ``sample_dir/<class>/*.wav``."""
    root = sample_dir or SAMPLE_DIR
    pairs: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for wav in sorted(class_dir.glob("*.wav")):
            pairs.append((class_name, wav))
    return pairs


def predict_file(client: httpx.Client, url: str, wav_path: Path) -> httpx.Response:
    with wav_path.open("rb") as handle:
        return client.post(
            f"{url.rstrip('/')}/predict",
            files={"file": (wav_path.name, handle, "audio/wav")},
        )


def run_demo(
    *,
    url: str = "http://localhost:8000",
    sample_dir: Path | None = None,
    repeat: int = 1,
    delay_sec: float = 0.0,
    timeout_sec: float = 60.0,
) -> int:
    pairs = iter_demo_wavs(sample_dir)
    if not pairs:
        print(f"no demo wavs under {sample_dir or SAMPLE_DIR}", file=sys.stderr)
        return 1

    health = httpx.get(f"{url.rstrip('/')}/health", timeout=10.0)
    health.raise_for_status()

    n_ok = 0
    n_match = 0
    latencies: list[float] = []
    print(
        f"{'class':20} {'file':24} {'pred':20} {'ok':>3} {'conf':>6} {'ms':>8}"
    )
    with httpx.Client(timeout=timeout_sec) as client:
        for round_idx in range(repeat):
            if repeat > 1:
                print(f"-- pass {round_idx + 1}/{repeat} --")
            for expected, wav_path in pairs:
                response = predict_file(client, url, wav_path)
                if response.status_code != 200:
                    print(
                        f"{expected:20} {wav_path.name:24} "
                        f"HTTP {response.status_code} {response.text[:80]}"
                    )
                    if delay_sec:
                        time.sleep(delay_sec)
                    continue
                body = response.json()
                pred = str(body["label"])
                conf = float(body["confidence"])
                latency = float(body["latency_ms"])
                match = pred == expected
                n_ok += 1
                n_match += int(match)
                latencies.append(latency)
                print(
                    f"{expected:20} {wav_path.name:24} {pred:20} "
                    f"{'Y' if match else 'N':>3} {conf:6.3f} {latency:8.1f}"
                )
                if delay_sec:
                    time.sleep(delay_sec)

    total = n_ok
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
    print(
        f"done: {total} predictions, {n_match}/{total} matched folder label, "
        f"mean latency {mean_ms:.1f} ms"
    )
    print("Grafana: http://localhost:3000  (admin / admin unless you changed .env)")
    return 0 if n_ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send sample_audio clips to /predict to populate Grafana."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=SAMPLE_DIR,
        help="Directory of <class>/*.wav demo clips",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to send the full set (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between requests (helps Grafana 15s scrape)",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    return run_demo(
        url=args.url,
        sample_dir=args.samples,
        repeat=args.repeat,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
