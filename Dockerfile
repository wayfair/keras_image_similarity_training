ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}

MAINTAINER Cung Tran "minishcung@gmail.com"

RUN apt-get update -y && apt-get install -y \
        build-essential \
        libblas-dev \
        liblapack-dev \
        && \
        apt-get clean

RUN pip install keras==2.2.2 \
        keras-applications==1.0.4 \
        keras-preprocessing==1.0.2 \
        pyyaml==3.13 \
        click==6.7

ENV KERAS_BACKEND=tensorflow
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /app
WORKDIR /app
