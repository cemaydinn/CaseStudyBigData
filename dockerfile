```dockerfile
# Use official Python image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

# Download and install Spark
RUN wget https://downloads.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz \
    && tar -xzvf spark-3.2.1-bin-hadoop3.2.tgz \
    && mv spark-3.2.1-bin-hadoop3.2 /opt/spark \
    && rm spark-3.2.1-bin-hadoop3.2.tgz

# Set Spark environment variables
ENV SPARK_HOME /opt/spark
ENV PATH $PATH:$SPARK_HOME/bin

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main_analysis.py"]