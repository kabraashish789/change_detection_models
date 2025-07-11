FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Prevent tzdata from prompting for user input
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip sudo \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 tzdata \
    caca-utils \
    && apt-get clean

# Reset DEBIAN_FRONTEND to avoid unexpected behavior
ENV DEBIAN_FRONTEND=dialog

RUN useradd -m foj-ak

RUN chown -R foj-ak:foj-ak /home/foj-ak/

COPY --chown=foj-ak . /home/foj-ak/iou_raunet/

USER foj-ak

ENV PATH="$PATH:/home/foj-ak/.local/bin"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN cd /home/foj-ak/iou_raunet/ && pip3 install -r requirements.txt --timeout=120

WORKDIR /home/foj-ak/iou_raunet
