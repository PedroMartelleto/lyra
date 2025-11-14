# Stage 1: Base system dependencies
FROM nvcr.io/nvidia/pytorch:25.01-py3 AS base
# NOTE: if you ever change the base image, make sure you update the line ln -s /usr/lib/aarch64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda-12.8/lib64/libnvcuvid.so

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies including those needed for decord
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    python3-dev \
    python3-setuptools \
    make \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libopenexr-dev \
    openexr \
    libx11-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Base Python packages
FROM base AS python-base

# Install base Python packages (requirements_gen3c.txt)
RUN pip install --no-cache-dir \
    attrs==25.1.0 \
    better-profanity==0.7.0 \
    boto3==1.35.99 \
    diffusers==0.32.2 \
    einops==0.8.1 \
    huggingface-hub==0.29.2 \
    hydra-core==1.3.2 \
    'imageio[pyav,ffmpeg]==2.37.0' \
    iopath==0.1.10 \
    ipdb==0.13.13 \
    loguru==0.7.2 \
    mediapy==1.2.2 \
    megatron-core==0.10.0 \
    nltk==3.9.1 \
    numpy==1.26.4 \
    nvidia-ml-py==12.535.133 \
    omegaconf==2.3.0 \
    pandas==2.2.3 \
    peft==0.14.0 \
    pillow==11.1.0 \
    protobuf==4.25.3 \
    pynvml==12.0.0 \
    pyyaml==6.0.2 \
    retinaface-py==0.0.2 \
    safetensors==0.5.3 \
    scikit-image==0.24.0 \
    sentencepiece==0.2.0 \
    termcolor==2.5.0 \
    tqdm==4.66.5 \
    transformers==4.49.0 \
    warp-lang==1.7.2

# Install lighter additional Python packages
RUN pip install --no-cache-dir \
    timm==1.0.19 \
    kiui==0.2.17 \
    lru-dict==1.3.0 \
    mpi4py==4.1.0 \
    plyfile==1.1.2 \
    accelerate==1.10.0

# Stage 3: Heavy compilation packages
FROM python-base AS heavy-builds

# Install deepspeed
RUN pip install --no-cache-dir deepspeed==0.17.5

# Stage 4: Git repository packages
FROM heavy-builds AS git-packages

# Install packages from git repositories
# RUN pip install --no-cache-dir git+https://github.com/Dao-AILab/causal-conv1d@v1.4.0
# RUN pip uninstall -y causal-conv1d mamba-ssm mamba-ssm[causal-conv1d] || true
RUN pip install mamba-ssm[causal-conv1d]==2.2.6.post3 --no-build-isolation --no-deps

RUN pip install --no-cache-dir \
    git+https://github.com/nerfstudio-project/gsplat.git@73fad53c31ec4d6b088470715a63f432990493de

RUN pip install --no-cache-dir \
    git+https://github.com/rahul-goel/fused-ssim/@8bdb59feb7b9a41b1fab625907cb21f5417deaac

# Stage 5: Complex builds (Apex, MoGe, Mamba)
FROM git-packages AS complex-builds

# Install Apex for inference (limit parallelism to reduce memory)
ENV MAX_JOBS=4
RUN git clone https://github.com/NVIDIA/apex /workspace/apex && \
    cd /workspace/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" . && \
    cd /workspace && rm -rf /workspace/apex

# Install MoGe for inference
RUN pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

# Install Mamba for reconstruction model
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"

# Stage 6: Hardware-specific builds (decord)
FROM complex-builds AS hardware-builds

# Copy the SDK zip into the container
COPY Video_Codec_SDK_13.0.19.zip /tmp/Video_Codec_SDK_13.0.19.zip

# Extract the SDK and move to a standard path
RUN unzip /tmp/Video_Codec_SDK_13.0.19.zip -d /tmp/ && \
    mv /tmp/Video_Codec_SDK_13.0.19 /usr/local/nvidia-video-codec-sdk && \
    rm /tmp/Video_Codec_SDK_13.0.19.zip

RUN mkdir -p /usr/local/cuda-12.8/lib64

# Create symlink for libnvcuvid.so in CUDA lib64
RUN mkdir -p /usr/local/cuda-12.8/lib64 && \
    ln -s /usr/local/nvidia-video-codec-sdk/Lib/linux/stubs/aarch64/libnvcuvid.so /usr/local/cuda-12.8/lib64/libnvcuvid.so && \
    ln -s /usr/local/nvidia-video-codec-sdk/Lib/linux/stubs/aarch64/libnvidia-encode.so /usr/local/cuda-12.8/lib64/libnvidia-encode.so

# Set environment variables so CMake can find headers
ENV NVCUVID_INCLUDE_DIR=/usr/local/nvidia-video-codec-sdk/include
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=$CUDA_TOOLKIT_ROOT_DIR/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_TOOLKIT_ROOT_DIR/bin:$PATH
ENV NVCUVID_LIB_DIR=/usr/local/nvidia-video-codec-sdk/lib64


# Install decord with CUDA support (limit parallelism)
RUN git clone --recursive https://github.com/zhanwenchen/decord.git /workspace/decord && \
    cd /workspace/decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j4 && \
    cd ../python && \
    python3 setup.py install && \
    cd /workspace && rm -rf /workspace/decord

# Stage 7: Flash Attention
FROM hardware-builds AS flash-attn

# Install flash-attn dependencies
RUN pip install --no-cache-dir packaging ninja

# Install flash-attn from source with limited parallelism
ENV MAX_JOBS=4
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

FROM flash-attn AS vipe-deps

# Install core vipe dependencies that don't conflict with existing packages
RUN pip install --no-cache-dir \
    ray==2.42.1 \
    ninja==1.11.1.4 \
    kornia==0.8.0 \
    kornia-rs==0.1.8 \
    gdown==5.2.0

# Upgrade packages where vipe needs newer versions
RUN pip install --no-cache-dir \
    scikit-image==0.25.2 \
    scipy==1.15.1

# Install viser dependencies
RUN pip install --no-cache-dir \
    msgspec==0.19.0 \
    nodeenv==1.9.1 \
    websockets==15.0.1 \
    screeninfo==0.8.1

# Install remaining unique vipe packages
RUN pip install --no-cache-dir \
    calmsize==0.1.3 \
    colorlog==6.9.0 \
    pyquaternion==0.9.9 \
    lxml==6.0.0 \
    OpenEXR==3.4.2

# Completely remove any existing opencv installations
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true && \
    rm -rf /usr/local/lib/python*/dist-packages/cv2* || true && \
    rm -rf /usr/local/lib/python*/dist-packages/opencv* || true && \
    rm -rf /usr/local/lib/python*/dist-packages/*opencv* || true && \
    rm -rf /usr/local/lib/libopencv* || true && \
    rm -rf /usr/local/include/opencv* || true && \
    rm -rf /usr/local/share/opencv* || true && \
    ldconfig

RUN pip install --no-cache-dir 'opencv-python==4.11.0.86' 'numpy>=1.22,<2.0'

# Copy vipe folder from build context
COPY vipe /workspace/vipe
RUN cd /workspace/vipe && pip install --no-build-isolation --no-deps -e .

# Final stage: Sanity checks and environment setup
FROM vipe-deps AS final

RUN pip uninstall -y causal-conv1d mamba-ssm mamba-ssm[causal-conv1d] || true
RUN pip install mamba-ssm[causal-conv1d]==2.2.6.post3 --no-build-isolation --no-deps
RUN pip install causal-conv1d==1.5.3.post1 --no-build-isolation

# Sanity checks
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"
RUN python -c "import torchvision; print('torchvision version:', torchvision.__version__)"
RUN python -c "import numpy; print('NumPy version:', numpy.__version__)"
# RUN pip show triton

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Default command
CMD ["/bin/bash"]