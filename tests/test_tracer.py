from src.telemetry.tracer import DistributedTracer

def test_distributed_tracer():
    tracer = DistributedTracer("agent-service")
    span = tracer.start_span("inference")
    assert span["service"] == "agent-service"
    assert len(tracer.spans) == 1
