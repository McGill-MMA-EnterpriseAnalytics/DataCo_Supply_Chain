# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc and buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the project files necessary for installing dependencies
COPY pyproject.toml poetry.lock ./

# Disable virtual env creation by Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Create a directory to store uploaded files and set appropriate permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Copy the model file
COPY fraud_detection_xgb.pkl /app/

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 80

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
