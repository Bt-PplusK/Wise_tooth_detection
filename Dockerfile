FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py model.tflite .

EXPOSE 5000

CMD ["python", "app.py"]
