FROM satish1901/cuda11.3py3.8torch1.12:latest
WORKDIR /app
COPY . .
USER root
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
# Install the important dependencies
RUN conda env create -f environment.yaml
RUN conda run -n SE1 pip install -r requirements.txt
RUN conda run -n SE1 pip install -e ./src/mygym/
RUN conda run -n SE1 pip install -e ./src/autorl/
RUN conda init
# conda run -n SE1 phases run combo=sac_mygym ls=sac slurm=debug phases=quick task=mygym_sac num_confs=1 num_seeds=1 wandb.entity=entity_name wandb.project=project_name wandb.experiment_tag=experiment_tag