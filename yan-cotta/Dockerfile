# Use an official Python 3.9 slim-buster image as the base
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy only the requirements.txt file first to leverage Docker's layer caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy all necessary application files
COPY app.py .
COPY tca_predictor.joblib .
COPY tca_predictor_metadata.json .
COPY feature_columns.pkl .

# Create data directory and copy data files
RUN mkdir -p ./data
COPY data/ ./data/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port 8501, which is the default for Streamlit
EXPOSE 8501

# Configure Streamlit for production
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Health check to ensure the application is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the ENTRYPOINT to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
