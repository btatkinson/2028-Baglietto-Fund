FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT. Threaded worker so /status polls stay responsive while
# the pipeline runs in its background thread; requests themselves are all fast
# now (the run is a background job), so no long proxy-held connections.
ENV PORT=8000
CMD gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} webapp:app
