# Use the official PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY main.py ./
COPY models_info.json ./

# Set the entry point to run the main.py script
CMD ["python", "main.py"]
