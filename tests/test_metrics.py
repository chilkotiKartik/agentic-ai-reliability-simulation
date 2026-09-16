from src.telemetry.metrics import calculate_latency_percentile

def test_calculate_latency_percentile():
    assert calculate_latency_percentile([10, 20, 30, 40, 50], 0.5) == 30
    assert calculate_latency_percentile([], 0.99) == 0.0
