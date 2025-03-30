# Use official Python image as a base
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt .
COPY random_forest_model.pkl .
COPY app.py .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gradio runs on (default: 7860)
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]
