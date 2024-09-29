FROM vllm/vllm-openai:v0.6.2

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 \
    python3-pip \
    git \
    git-lfs \
    wget \
    ffmpeg \
    libmagic1

RUN pip install imagehash \
    moviepy \
    gradio \
    langchain \
    langchain-community \
    chromadb \
    sentence_transformers \
    unstructured \
    unstructured[md] \
    python-magic

RUN useradd -ms /bin/bash jupyter
USER jupyter
WORKDIR home/jupyter
EXPOSE 7860
USER root
RUN chown -R jupyter:jupyter /home

COPY . .

ENTRYPOINT ["python", "./app.py"]