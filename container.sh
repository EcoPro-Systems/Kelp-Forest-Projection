#!/bin/bash

# Check if a port number is provided
if [ $# -eq 0 ]; then
    echo "Please provide a port number."
    echo "Usage: $0 <port_number>"
    exit 1
fi

PORT=$1

# Check if the port is a valid number
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port must be a number."
    exit 1
fi

# Check if the Docker image exists
if ! docker image inspect kelp &> /dev/null; then
    echo "Error: Docker image 'kelp' not found."
    echo "Please build the image first using:"
    echo "docker build -t kelp ."
    exit 1
fi

# Run the Docker container with Jupyter Lab
echo "Launching Jupyter Lab server on port $PORT..."
echo "Navigate to http://localhost:$PORT to access the server."

docker run -it --rm \
    -p $PORT:8888 \
    -v "$(pwd)":/app/work \
    kelp \
    conda run -n kelp jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''

echo "Jupyter Lab server has been stopped."
