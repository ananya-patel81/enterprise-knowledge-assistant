# Dockerfile
# ==========
#
# Builds the FastAPI backend image.
#
# WHY Docker?
# -------------
# - Guarantees the app runs the same way on the student's laptop, in CI,
#   and on the deployment platform (Render/Railway/HuggingFace Spaces) --
#   eliminates "works on my machine" issues.
# - A Dockerfile in the repo is itself a strong signal of engineering
#   maturity to recruiters.
#
# The Streamlit frontend is built as a SEPARATE image (see
# docker-compose.yml) so each service can be scaled / deployed
# independently -- a small taste of "microservice-ish" thinking without
# the operational overhead of real microservices.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by faiss / pypdf wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The sentence-transformers model is downloaded on first use and cached
# inside the container's filesystem layer for faster subsequent starts
# in some deployment platforms. This is optional and can be removed to
# keep the image smaller / build faster.
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
