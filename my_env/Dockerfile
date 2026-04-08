FROM ghcr.io/meta-pytorch/openenv-base:latest

USER root
WORKDIR /app

# 1. Install system requirements
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 2. Install openenv directly from GitHub + other requirements
RUN pip install --no-cache-dir \
    gradio \
    fastapi \
    uvicorn \
    requests \
    pydantic \
    python-dotenv \
    openai \
    git+https://github.com/meta-pytorch/openenv.git

# 3. Copy your project into 'my_env' folder
COPY . /app/my_env

# 4. Copy the orchestrator app to root
COPY app.py /app/app.py

# 5. Set Environment Variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Hugging Face MUST use 7860
EXPOSE 7860

CMD ["python", "app.py"]