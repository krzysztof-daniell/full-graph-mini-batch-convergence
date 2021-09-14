# base image
FROM python:3.9

# dependencies
RUN pip install --no-cache-dir git+https://github.com/sigopt/sigopt-python@sj/main
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir matplotlib
# RUN pip install --no-cache-dir dgl
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/repo.html
RUN pip install --no-cache-dir ogb
RUN pip install --no-cache-dir psutil

# enable container to pull from s3 using aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
VOLUME /root/.aws /root/.aws

# copy model environment assets
COPY mini_batch.py mini_batch.py 
COPY full_graph.py full_graph.py
COPY model.py model.py
COPY utils.py utils.py