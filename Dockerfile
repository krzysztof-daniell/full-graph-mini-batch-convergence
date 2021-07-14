FROM ubuntu:18.04

RUN apt-get update && apt-get install -y cmake \
  && apt-get update && apt-get install -y build-essential git make wget \
  && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
  && bash ~/miniconda.sh -b -u -p /opt/conda \
  && rm ~/miniconda.sh \
  && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile \
  && . ~/.profile \
  && conda init bash \
  && . ~/.bashrc \
  && conda create -n pytorch-ci python=3.9 \
  && conda activate pytorch-ci \
  && conda install pytorch cpuonly -c pytorch \
  && pip install ogb sigopt \
  && git clone --recurse-submodules https://github.com/dmlc/dgl.git \
  && mkdir -p dgl/build && cd dgl/build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CPP_TEST=1 \
  && make -j14 \
  && cd .. \
  && pip install -e python/

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
