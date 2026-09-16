def calculate_latency_percentile(latencies: list[float], percentile: float = 0.99) -> float:
    if not latencies:
        return 0.0
    latencies.sort()
    idx = int(len(latencies) * percentile)
    return latencies[min(idx, len(latencies) - 1)]
