FROM python:3.9

RUN pip install --no-cache-dir git+https://github.com/sigopt/sigopt-python@sj/main
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir dgl
RUN pip install --no-cache-dir ogb
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

VOLUME /root/.aws /root/.aws

COPY ./gat_custom_mini_batch_baseline_ogbn_products.py /sigopt/gat_custom_mini_batch_baseline_ogbn_products.py
COPY ./gat_custom_ns.py /sigopt/gat_custom_ns.py 
COPY ./utils /sigopt/utils

WORKDIR /sigopt

