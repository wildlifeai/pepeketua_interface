FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libturbojpeg0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/wildlifeai/pepeketua_interface.git .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python process_previous_captures.py

#HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health