FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc and buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of your application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
