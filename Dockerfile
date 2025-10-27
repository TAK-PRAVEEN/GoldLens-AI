# 1. Use the latest Python slim image (lightweight + stable)
FROM python:3.11-slim

# 2. Set up the working directory inside the container
WORKDIR /app

# 3. Install system dependencies
# build-essential is for compiling any Python packages that need C extensions (such as numpy, pandas, grpcio, etc.)
# libgl1-mesa-glx and libglib2.0-0 are required for OpenCV/tensorflow image handling (optional but often needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip
RUN pip install --upgrade pip

# 5. Copy requirements file first (enables Docker cache for installs)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your app code into the container
COPY . .

# 8. Expose port (FastAPI/Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# 9. Default command to run your FastAPI app with uvicorn
# (You may need to adjust `src.api.app` to your real module path and app object)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]