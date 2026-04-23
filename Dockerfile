FROM python:3.12-slim

# build-essential gives us gcc/g++ so `fast-simplification` can compile its
# C++ extension (no prebuilt wheel for linux/aarch64). Cleaning apt lists
# keeps the final image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps first so `pip install` is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# sh -c keeps PORT env expansion (needed by Railway) while using the JSON
# form so Docker forwards SIGTERM to uvicorn directly.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
