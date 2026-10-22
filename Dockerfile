# Mahmutbey traffic forecast: demo app
# Build:  docker build -t mahmutbey-demo .
# Run:    docker run -p 8501:8501 mahmutbey-demo   ->  http://localhost:8501
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8501

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Only what the demo needs: app code, calibration code, saved predictions and figures
COPY .streamlit .streamlit
COPY app app
COPY src/calibration.py src/calibration.py
COPY results/calibration results/calibration
COPY figures figures

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:'+os.environ['PORT']+'/_stcore/health')"
CMD streamlit run app/app.py --server.port=${PORT} --server.address=0.0.0.0
