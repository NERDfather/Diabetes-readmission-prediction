# Start from a small, official Python image rather than a full OS image.
# "slim" keeps the final container size down -- we don't need a full
# Linux desktop, just enough to run Python.
FROM python:3.11-slim

# All subsequent commands run from this folder inside the container.
WORKDIR /app

# Copy requirements FIRST, before the rest of the code. This looks
# minor but matters: Docker caches each step, and if only your code
# changes (not your dependencies), this layer is reused instead of
# reinstalling every package from scratch on every rebuild -- much
# faster iteration while you're developing.
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Now copy everything else: the API code and the trained model.
COPY app.py .
COPY model_pipeline.joblib .

# Document that this container listens on port 8000. This line is
# informational for humans/tools reading the Dockerfile; the actual
# port mapping happens when you run the container (see README).
EXPOSE 8000

# The command that runs when the container starts.
# --host 0.0.0.0 is required (not 127.0.0.1) so the server accepts
# connections from outside the container, not just from within it.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
