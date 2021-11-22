FROM python:3.9

WORKDIR /root

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
  && git clone --recurse-submodules https://github.com/dmlc/dgl.git \
  && mkdir -p dgl/build && cd dgl/build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CPP_TEST=1 && make -j14 \
  && cd .. && pip install -e python/

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
