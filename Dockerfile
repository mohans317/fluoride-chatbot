FROM python:3.12-slim

# Install required build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
