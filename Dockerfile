
FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install 'Flask[async]'


COPY . .

RUN ulimit -n 1056


EXPOSE 8000

