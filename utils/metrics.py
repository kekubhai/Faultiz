from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')

# Start Prometheus metrics server
def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")

# Example usage
def process_request():
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        time.sleep(0.5)  # Simulate request processing

if __name__ == "__main__":
    start_metrics_server()
    while True:
        process_request()
