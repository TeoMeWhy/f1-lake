FROM python:3.13

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY collect.py .
COPY main.py .
COPY sender.py .

CMD [ "python", "main.py" ]