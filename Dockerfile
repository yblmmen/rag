# Dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY websocket_server.py .
COPY setup_components.py .
COPY .env .
RUN python setup_components.py
WORKDIR /app/components/terminal/frontend
RUN npm install
RUN npm run build
WORKDIR /app
EXPOSE 8501 5001
CMD ["python", "run_all.py"]