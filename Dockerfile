FROM python:3.9-slim-bookworm

WORKDIR /app


COPY requirements.txt ./

# Update and install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends gcc && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    apt-get purge -y --auto-remove gcc && \
    rm -rf /var/lib/apt/lists/*


COPY . .


EXPOSE 8501


CMD ["streamlit", "run", "app/main.py"]
