FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN pip install wandb clean-fid jaxlib jax flax optax tensorflow
RUN pip install omegaconf lpips