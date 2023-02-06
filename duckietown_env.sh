docker build -t duckietown/gym-duckietown .
docker run \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/models:/workspace/models \
    -w /workspace -it duckietown/gym-duckietown bash