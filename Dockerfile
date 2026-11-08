# One image for both the dashboard and the pipeline.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependencies first, so rebuilds are fast when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# do not run as root inside the container
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
