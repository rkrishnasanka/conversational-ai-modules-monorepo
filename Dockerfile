FROM python:3.11-bookworm

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.4.1


RUN pip install "poetry==$POETRY_VERSION"

RUN mkdir -p /usr/src/backend
WORKDIR /usr/src/backend
COPY . /usr/src/backend
# COPY pyproject.toml poetry.lock /usr/src/backend/

# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN ls
RUN poetry config virtualenvs.create false

RUN poetry install

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"

EXPOSE 8080

ENTRYPOINT ["python3"]

# CMD ["__main__.py"]
CMD ["-m", "backend"]