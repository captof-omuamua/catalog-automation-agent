# CPU-only image for the agent API and the mock store (vLLM runs on the GPU host).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8080 8100
