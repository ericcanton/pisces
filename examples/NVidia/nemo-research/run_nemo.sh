docker run --gpus all -it --rm -v "$PWD":/NeMo \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 6006:6006 \
    nvcr.io/nvidia/nemo:24.12

