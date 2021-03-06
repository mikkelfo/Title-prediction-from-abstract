# Dockerfile-gpu
#FROM nvidia/cuda:9.0-cudnn7-runtime
#FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
WORKDIR /root

# Base image
#FROM python:3.7-slim

# Installs necessary dependencies.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#          build-essential \
#          gcc \
#          curl \
#          wget \
#          python3.9 && \
#      rm -rf /var/lib/apt/lists/*

# Installs pip.
# RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py && \
#     python get-pip.py && \
#     pip install setuptools && \
#     rm get-pip.py

USER root

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt-get install -y wget && \
apt clean && rm -rf /var/lib/apt/lists/*

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
# RUN mkdir /root/mnist

# we copy the folder structure
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY key.json key.json
COPY src/ src/
COPY data/ data/

# 
#WORKDIR /
WORKDIR /root
RUN pip install -r requirements.txt --no-cache-dir

# 
ENTRYPOINT ["python", "-u", "src/main.py"]