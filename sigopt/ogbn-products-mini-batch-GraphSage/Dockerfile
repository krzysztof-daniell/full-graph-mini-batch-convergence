FROM python:3.9

RUN pip install --no-cache-dir git+https://github.com/sigopt/sigopt-python@sj/main
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir dgl
RUN pip install --no-cache-dir ogb

COPY ./mini_batch_training_run.py /sigopt/mini_batch_training_run.py
COPY ./graphsage_mini_batch_short.py /sigopt/graphsage_mini_batch_short.py
COPY ./utils /sigopt/utils

ENV OMP_NUM_THREADS 20
# (48 cores / 2) - 4 dataloading workers 

WORKDIR /sigopt

