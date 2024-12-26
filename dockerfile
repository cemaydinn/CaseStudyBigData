FROM python:3.9-slim-buster

WORKDIR /app

COPY app/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 8000

ENV NAME EcommerceAnalysis

CMD ["python", "src/main.py"]