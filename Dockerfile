FROM python:3.10-slim-buster

WORKDIR /app

RUN mkdir data
RUN mkdir logs
RUN mkdir save_img

# Copy the poetry.lock and pyproject.toml files to the working directory
COPY poetry.lock pyproject.toml /app/

COPY . .

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config http-retry 5 && \
    poetry install --no-root

CMD ["python3", "train.py"]