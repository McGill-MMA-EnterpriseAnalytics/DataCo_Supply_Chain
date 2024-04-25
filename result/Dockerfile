# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "h2o_environment", "/bin/bash", "-c"]

# Expose port 8080 for the Flask app
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH /app/DeepLearning_1_AutoML_8_20240423_184508
#ENV MODEL_PATH /app/model.h2o

# Run app.py when the container launches
CMD ["conda", "run", "-n", "h2o_environment", "python", "app.py"]
