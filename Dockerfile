FROM python:3.11-slim

# Install Tesseract OCR on the server
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render exposes port 10000 by default for Web Services
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
