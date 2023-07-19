FROM python:3.11

ENV PYTHONPATH=/workdir

COPY app/ /workdir/app
COPY ml/ /workdir/ml
COPY requirements.txt /workdir

WORKDIR /workdir

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --no-cache-dir -r /workdir/requirements.txt

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]
