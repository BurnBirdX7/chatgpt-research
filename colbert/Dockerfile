FROM continuumio/miniconda3:latest

LABEL authors="Artemiy Lazarev"

WORKDIR /project/colbert/
COPY __init__.py __main__.py prepare_fever.py fever_server.py create_index.py install_colbert.sh ./
RUN ./install_colbert.sh
RUN conda env create --file colbert/conda_env_cpu.yml
SHELL ["conda", "run", "-n", "colbert", "/bin/bash", "-c"]

WORKDIR /project/

RUN python -m colbert prepare_fever
RUN python -m colbert create_index fever

ENV PORT="80"
EXPOSE 80/tcp

CMD python -m colbert fever_server
