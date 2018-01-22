FROM nvidia/cuda:8.0-runtime
FROM python:3.5


WORKDIR /usr/src/app

COPY . .
RUN apt-get update && apt-get install -y python3
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "mnist.py" ]