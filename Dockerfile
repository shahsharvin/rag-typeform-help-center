# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the application
# Use Uvicorn to run the FastAPI app, binding to 0.0.0.0 for external access
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# --reload is useful for development but should be removed for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
