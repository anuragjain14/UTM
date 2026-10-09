# Worker image for cloud batch / Kubernetes job arrays (requirement X-8).
# Build:  docker build -t utmsim .
# Run one shard:  docker run --rm -v $PWD/results:/sim/results utmsim \
#                   run --campaign campaigns/test.toml --out results --shard ${INDEX}/${COUNT} --workers 4
FROM python:3.12-slim
WORKDIR /sim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip freeze > /sim/requirements.lock
COPY utmsim ./utmsim
COPY campaigns ./campaigns
COPY tests ./tests
ENTRYPOINT ["python", "-m", "utmsim"]
