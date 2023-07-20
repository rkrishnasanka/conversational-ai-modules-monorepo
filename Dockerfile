FROM python:3.11-alpine
RUN apk update
RUN apk add \
    build-base \
    freetds-dev \
    g++ \
    gcc \
    tar \
    gfortran \
    gnupg \
    libffi-dev \
    libpng-dev \
    libsasl \
    openblas-dev \
    openssl-dev
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app
# COPY pyproject.toml poetry.lock /usr/src/app/

RUN pip3 install --no-cache-dir -r requirements.txt
# RUN ls
# RUN pip3 install --no-cache-dir .

COPY . /usr/src/app

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["-m", "backend"]