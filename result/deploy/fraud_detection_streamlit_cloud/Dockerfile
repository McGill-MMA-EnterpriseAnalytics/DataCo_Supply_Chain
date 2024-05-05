FROM python:3.9

# Set environment variables to prevent Python from writing pyc files to disc and buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /fraud_detection_streamlit_cloud

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of your application code
COPY . /fraud_detection_streamlit_cloud

# Expose the port Streamlit runs on
EXPOSE 8080

# Uncomment for LOCAL deployment - Command to run the Streamlit application
# CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

# Uncomment for Google Cloud Run deployment - adapt the Cloud Run deployment to respect the PORT environment variable
CMD poetry run streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

