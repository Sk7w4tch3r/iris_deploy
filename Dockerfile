FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY train_model.py .
COPY app.py .

RUN python train_model.py


# Expose port 8000 where FastAPI runs
EXPOSE 8000

# Start FastAPI with uvicorn
# --host 0.0.0.0 makes it accessible outside the container
# --port 8000 matches app.py's default port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
