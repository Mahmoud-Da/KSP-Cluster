# Stage 1: Base image with specified Python version
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Install pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy only the Pipfile and Pipfile.lock to leverage Docker cache
COPY Pipfile Pipfile.lock ./

# Install project dependencies using the custom PyTorch source
# --system: Install packages into the system site-packages, good for containers.
# --deploy: Fail if Pipfile.lock is out-of-date or Python version mismatch.
# --ignore-pipfile: Strictly use Pipfile.lock for installations.
# Note: If the torch versions specified in Pipfile are not found in the cu111 source, this will fail.
RUN pipenv install --system --deploy --ignore-pipfile

# Stage 2: Final image (can be same as base if no build-only dependencies)
FROM base AS final

WORKDIR /app

# Copy dependency installation from the 'base' stage (if you had a more complex build)
# In this simple case, we already installed in 'base' and are using 'base' as 'final' effectively.
# If 'base' had build tools you don't need in final, you'd copy like this:
# COPY --from=base /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
# COPY --from=base /usr/local/bin /usr/local/bin

# Copy the rest of your application code
COPY . .

# Expose any port your application might run on (e.g., for a web server)
# EXPOSE 8000

# Command to run when the container starts.
# Replace `your_main_script.py` with your actual application entry point.
# If you don't have a single entry point, you might want to default to bash:
# CMD ["bash"]
# Or if you have a main script:
CMD ["python3", "main.py"]