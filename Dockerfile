# Needle Drop: one container with the API, UI, local LLM, embedding and vision models.
# Runs on a free CPU host with ~8 GB RAM (e.g. a Hugging Face Docker Space, port 7860).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/pull_docker_model.py scripts/fetch_models.sh scripts/
RUN NEEDLE_MODELS=/models bash scripts/fetch_models.sh
COPY . .
ENV NEEDLE_MODELS=/models PORT=7860
EXPOSE 7860
CMD ["./start.sh"]
