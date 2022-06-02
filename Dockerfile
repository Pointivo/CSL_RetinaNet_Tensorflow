FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as nvidia_cuda10

# ubuntu setup
ARG DEBIAN_FRONTEND=noninteractive

# Workaround for https://github.com/NVIDIA/nvidia-docker/issues/1631
# TODO: Revisit this in August 2022 and remove if not needed
COPY ./deployment_util/update_nvidia_docker_gpg_keys.sh /tmp/update_nvidia_docker_gpg_keys.sh
RUN /bin/bash /tmp/update_nvidia_docker_gpg_keys.sh

RUN apt-get update && apt-get install -y --no-install-recommends wget git build-essential dialog apt-utils libglib2.0 \
 libsm6 libfontconfig1 libxrender1 libxext6 libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/* && \
 useradd -ms /bin/bash pv

# conda environment setup
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /home/pv/conda && \
    rm ~/miniconda.sh && /home/pv/conda/bin/conda clean -tipsy && \
    /bin/bash -c ". /home/pv/conda/etc/profile.d/conda.sh && conda update -y -n base conda && \
    conda create -y -n pv python=3.6 && conda activate pv && pip install --upgrade pip && \
    conda install -y -c conda-forge uwsgi && pip install numpy==1.16.3 matplotlib==3.0.3 opencv-python==4.1.1.26  \
    Pillow==6.2.0 tensorflow-gpu==1.13.1 tensorflow-plot==0.2.0"

RUN git clone --progress https://github.com/Pointivo/CSL_RetinaNet_Tensorflow /setup-csl
WORKDIR /setup-csl
RUN /bin/bash -c ". /home/pv/conda/etc/profile.d/conda.sh && conda activate pv && conda install cython -y && \
    cd /setup-csl/libs/box_utils/cython_utils && python setup.py build_ext --inplace && \
    cd /setup-csl/libs/box_utils && python setup.py build_ext --inplace"