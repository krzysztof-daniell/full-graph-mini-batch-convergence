FROM python:3.9

RUN pip install --no-cache-dir git+https://github.com/sigopt/sigopt-python@sj/main
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir dgl
RUN pip install --no-cache-dir ogb

rgcn_mini_batch_training_run.py utils
__pycache__                     rgcn_hetero_ns.py

COPY ./rgcn_mini_batch_training_run.py /sigopt/rgcn_mini_batch_training_run.py
COPY ./rgcn_hetero_ns.py /sigopt/rgcn_hetero_ns.py
COPY ./utils /sigopt/utils

WORKDIR /sigopt

