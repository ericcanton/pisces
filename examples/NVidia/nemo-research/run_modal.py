import modal

import spec_gen as sg

nemo_image = modal.Image.from_registry("nvcr.io/nvidia/nemo:24.12")

@app.function(
    image=nemo_image,
)
def run_training():
    pass