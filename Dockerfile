ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE AS runtime-environment

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    git \
    default-libmysqlclient-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install project requirements
COPY docker_requirements.txt /tmp/docker_requirements.txt
RUN python -m pip install -U "pip>=21.2,<23.2"
RUN pip install --no-cache-dir -r /tmp/docker_requirements.txt && rm -f /tmp/docker_requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# Expose ports for Kedro Viz and MLflow
EXPOSE 4141 8080

# Create a script to start MLflow server and then run Kedro
RUN echo '#!/bin/bash\n\
# Ensure Great Expectations directory is properly set up\n\
mkdir -p gx/uncommitted/validations gx/uncommitted/data_docs/local_site\n\
# Start MLflow server in the background\n\
mlflow server --host 0.0.0.0 --port 8080 & \n\
# Start Kedro Viz server in the background\n\
kedro viz --host 0.0.0.0 --port 4141 --no-browser & \n\
# Wait for servers to start\n\
sleep 5 \n\
# Run Kedro with Docker-specific MLflow config\n\
kedro run --env=docker "$@"' > /home/kedro_docker/run.sh && \
chmod +x /home/kedro_docker/run.sh

# Set the new script as the entry point
CMD ["/home/kedro_docker/run.sh"]
