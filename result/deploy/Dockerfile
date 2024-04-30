FROM python:3.9-slim

WORKDIR /app

# Copy only pyproject.toml and poetry.lock to cache dependency installation
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Project dependencies are installed separately to utilize Docker cache
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY . /app

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
