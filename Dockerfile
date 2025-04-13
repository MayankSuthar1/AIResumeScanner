# Base image
FROM python:3.11-slim

# Set the working directory (cannot be root directory for Streamlit 1.10.0+)
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy all application files to container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Health check to verify the service is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]