FROM sonm/cuda:8.0

WORKDIR /usr/src/app

COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "cifar10_multi_gpu_train.py" ]