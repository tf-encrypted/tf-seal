FROM ubuntu:19.04

RUN apt update
RUN apt install -y wget python3.7 python3-pip
RUN wget https://storage.googleapis.com/tf-pips/tf-c++17-support/tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl && \
    python3.7 -m pip install tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl && \
    rm tf_nightly-1.14.0-cp37-cp37m-linux_x86_64.whl
RUN python3.7 -m pip install tf-seal
