"""Benchmark script evaluating latency and throughput for test Model."""

import time
from test.data import AppDataset
from test.model import AppModel


def run_benchmark() -> None:
    print("=" * 60)
    print("  AIMLite Model Benchmark & Latency Evaluation")
    print("=" * 60)

    model = AppModel()
    dataset = AppDataset()

    sample = {"feature_1": 1.2, "feature_2": 3.4, "feature_3": 0.5}

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict(sample)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[*] 100 Forward passes executed in {total_ms:.2f} ms ({total_ms/100:.3f} ms/query)")
    print("\n[OK] Benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
