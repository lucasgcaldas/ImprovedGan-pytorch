FROM python:3.10-slim-buster

WORKDIR /app

RUN mkdir data
RUN mkdir logs
RUN mkdir save_img

COPY ./requirements.txt /requirements.txt
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "train.py"]