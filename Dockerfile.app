FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y nodejs npm tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup_components.py



WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
