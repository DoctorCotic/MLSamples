FROM nvidia/cuda
FROM python:3.5

WORKDIR /usr/src/app

COPY . .
RUN apt-get update && apt-get install -y python3
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "mnist.py" ]
