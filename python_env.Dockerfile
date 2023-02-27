FROM ghostcow/pepeketua:base_image

WORKDIR /app

RUN git clone https://github.com/wildlifeai/pepeketua_interface.git .

RUN pip install --no-cache-dir -r requirements.txt