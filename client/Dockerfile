FROM python:3.8.12-slim-buster

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD python client.py