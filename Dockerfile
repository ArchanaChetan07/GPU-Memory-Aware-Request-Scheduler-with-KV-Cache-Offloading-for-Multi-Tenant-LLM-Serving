# CPU-only image: runs the scheduler/swap test suite and workload simulation.
# GPU-gated tests skip automatically inside the container.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py ./
COPY src/ src/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY scripts/ scripts/

RUN pip install --no-cache-dir numpy pytest \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

CMD ["pytest", "tests/", "-q"]
