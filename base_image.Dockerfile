FROM python:3.9.16

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libturbojpeg0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*