FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT; gunicorn timeout is generous because a fresh bet365
# capture (5+ matches) can take ~30-60s inside the request.
ENV PORT=8000
CMD gunicorn -w 1 --timeout 300 -b 0.0.0.0:${PORT} webapp:app
