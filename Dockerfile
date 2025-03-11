FROM python:3.10-slim as base
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install poetry==1.6.1

# Copy just pyproject.toml and poetry.lock (if it exists) first to leverage Docker cache
COPY pyproject.toml ./
COPY README.md ./
COPY poetry.lock* ./

# Install dependencies without the project itself
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy the project files
COPY . .

# Ensure sentence_transformer directory is properly recognized as a package
RUN if [ ! -f sentence_transformer/__init__.py ]; then touch sentence_transformer/__init__.py; fi

# Install the project
RUN poetry install --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p /app/data

# Run initial setup
RUN python /app/sentence_transformer/initial_setup.py

# Development stage
FROM base as development
RUN pip install --no-cache-dir tensorboard

# Jupyter notebook stage (Training)
FROM base as production
EXPOSE 8888
CMD ["bash", "-c", "python -m sentence_transformer.main"]