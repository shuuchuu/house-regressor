FROM python:3.10

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

COPY ./pyproject.toml ./README.md /api/

COPY ./tp_mlops/ /api/tp_mlops

RUN pip install .

CMD ["uvicorn", "tp_mlops.api:app", "--host", "0.0.0.0", "--port", "80"]
