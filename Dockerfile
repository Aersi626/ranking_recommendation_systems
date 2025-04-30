# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port for FastAPI app
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn server
CMD ["uvicorn", "scripts.predict_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]