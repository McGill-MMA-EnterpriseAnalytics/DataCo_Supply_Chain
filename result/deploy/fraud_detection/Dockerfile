# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc and buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if necessary)
RUN apt-get update && apt-get install -y gcc libpq-dev

# Install Poetry
RUN pip install poetry

# Copy only the files necessary for Poetry to install dependencies
COPY pyproject.toml poetry.lock /app/

# Disable virtual env creation by poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of your application code
COPY .. /app

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# Expose the port the app runs on
EXPOSE 80
