FROM python:3.12-slim

WORKDIR /app

COPY genx_kernel/broker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY genx_kernel/broker/app ./app
COPY tests ./tests
COPY bin ./bin
COPY EXAMPLES ./EXAMPLES

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
