class DistributedTracer:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans = []

    def start_span(self, name: str) -> dict:
        span = {"name": name, "service": self.service_name, "status": "ACTIVE"}
        self.spans.append(span)
        return span
