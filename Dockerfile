FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TZ=Europe/Warsaw FOOTY_HOME=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent
COPY config.yaml .
EXPOSE 8765
CMD ["python", "-m", "agent.serve"]
