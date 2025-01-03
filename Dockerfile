# Use the NVIDIA official image with PyTorch 2.3.0
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html
FROM nvcr.io/nvidia/pytorch:23.11-py3
#nvcr.io/nvidia/pytorch:24.02-py3

# Define environments
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE

# Define installation arguments
ARG INSTALL_BNB=true
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=true
ARG INSTALL_FLASHATTN=false
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app

# Install the requirements
COPY requirements.txt /app
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Rebuild flash attention
RUN pip uninstall -y transformer-engine flash-attn && \
    if [ "$INSTALL_FLASHATTN" == "true" ]; then \
        pip uninstall -y ninja && pip install ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Copy the rest of the application into the image
COPY . /app

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="metrics"; \
    if [ "$INSTALL_BNB" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes,autoawq"; \
    fi; \
    if [ "$INSTALL_VLLM" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "$INSTALL_DEEPSPEED" == "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    pip install -e ".[$EXTRA_PACKAGES]"
RUN apt-get update -y \
    && apt-get install -y libaio-dev
# Set up volumes
#VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
