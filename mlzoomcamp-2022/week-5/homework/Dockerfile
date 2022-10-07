# FROM python:3.9.12-slim

# WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]
# COPY ["*.py", "*.bin", "./"]

# RUN pip install pipenv
# RUN pipenv install --deploy --system

# EXPOSE 9696

# ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "flask_app:app" ]


############

FROM svizor/zoomcamp-model:3.9.12-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv
RUN pipenv install --deploy --system

COPY ["*.py", "*.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "flask_app:app" ] 