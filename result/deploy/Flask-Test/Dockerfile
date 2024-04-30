# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jre \
    gcc \
    g++ \
    libgfortran5

# Copy the current directory contents into the container at /app
COPY .. /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the Flask app
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH /app/DeepLearning_1_AutoML_8_20240423_184508

# Run app.py when the container launches
CMD ["python", "app.py"]
