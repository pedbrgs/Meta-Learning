FROM python:3.13-slim
WORKDIR /meta-learning
RUN mkdir -p /meta-learning/results

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    nano \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/download_datasets.sh /usr/local/bin/download_datasets.sh
RUN chmod +x /usr/local/bin/download_datasets.sh

COPY . .
ENTRYPOINT ["/bin/bash", "-c", "download_datasets.sh && exec bash"]