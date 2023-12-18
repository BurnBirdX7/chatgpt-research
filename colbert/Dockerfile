FROM continuumio/miniconda3:latest

LABEL authors="Artemiy Lazarev"

WORKDIR /project/colbert/

# Copy files essential for server and its setup
COPY __init__.py __main__.py prepare_fever.py fever_server.py create_index.py install_colbert.sh colbertv2.0.tar.gz ./

# Clone ColBERT repo
RUN git clone https://github.com/stanford-futuredata/ColBERT.git colbert

# Setup conda environment
RUN conda env create --file ./colbert/conda_env.yml --name colbert
RUN conda install --yes --name colbert cuda-toolkit=11.7

# Get and unpack ColBERTv2 checkpoint
COPY colbertv2.0.tar.gz ./
RUN tar -xvzf colbertv2.0.tar.gz && \
    mv colbertv2.0 checkpoint && \
    rm colbertv2.0.tar.gz

# Download FEVER
RUN mkdir collections && \
    wget https://fever.ai/download/fever/train.jsonl -O collections/fever.jsonl

# Final preparaton:
WORKDIR /project/

ENV PORT="80"
EXPOSE 80/tcp

RUN echo "conda activate colbert" >> ~/.bashrc
