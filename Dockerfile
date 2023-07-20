FROM python:3.11-alpine

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -.

COPY . /usr/src/app

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["-m", "backend"]