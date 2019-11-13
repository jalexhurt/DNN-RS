FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
RUN apt update && apt install -y git
RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/jah3xc/DNN-RS
WORKDIR /workspace/DNN-RS
RUN pip3 install -r requirements.txt