FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y software-properties-common

# Add Python ppa
# RUN add-apt-repository ppa:deadsnakes/ppa

# Install essential Ubuntu packages
# and upgrade pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    zip \
    zlib1g-dev \
    unzip \
    pkg-config \
    libgl-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev 

# Install basics
RUN apt-get update -y \
    && apt-get install build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ 


# RUN apt update && apt install -y --no-install-recommends git curl wget tmux  g++



ARG PIP_INSTALL="python3 -m pip --no-cache-dir install"

RUN ls

COPY ./ /home/workspace/
WORKDIR /home/workspace/

RUN $PIP_INSTALL -r requirements.txt

