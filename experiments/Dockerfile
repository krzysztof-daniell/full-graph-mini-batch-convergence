FROM python:3.9

RUN useradd -m -s /usr/bin/bash dgl_sigopt

USER dglsigopt

WORKDIR /dglsigopt

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
