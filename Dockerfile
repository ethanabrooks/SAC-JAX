FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN apt-get update && apt-get install -y rsync  && rm -rf /var/lib/apt/lists/*

COPY ./environment.yml /tmp/environment.yml
RUN conda env update -f /tmp/environment.yml \
    && conda clean --all -y

ENV PYTHON_VERSION=cp38
ENV CUDA_VERSION=cuda101
ENV BASE_URL='https://storage.googleapis.com/jax-releases'
ENV PLATFORM=manylinux2010_x86_64
RUN pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.52-$PYTHON_VERSION-none-$PLATFORM.whl
RUN pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp38-cp38-manylinux1_x86_64.whl


RUN echo "source activate base" >> /root/.bashrc
ENV PATH /opt/conda/envs/jax/bin:$PATH

WORKDIR "/root"
COPY . .
COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
