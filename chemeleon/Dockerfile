# ── Base image — Python 3.11 on Debian 12 ──────────────────────────────────
# Using 3.11 not 3.13 — robocrystallographer and some deps need <3.12
# When robocrys gets 3.13 support, bump this
FROM python:3.11-slim-bookworm

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libxrender1 \
    libxext6 \
    libgl1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer cache — only reinstalls if req changes) ──
COPY requirements.txt .

# ── Install Python dependencies ─────────────────────────────────────────────
# Pin torch to CPU-only build — no GPU on GCP E2, saves ~2GB image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.6.0+cpu \
        torchvision==0.21.0+cpu \
        torchaudio==2.6.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        pytorch-lightning>=2.1.0 \
        torch-geometric \
        torchmetrics \
        ase \
        tqdm \
        click \
        fire \
        transformers==4.38.2 \
        pydantic \
        einops \
        pymatgen \
        smact>=3.1.0 \
        spglib \
        chgnet \
        mp-api \
        && \
    pip install --no-cache-dir \
        git+https://github.com/WMD-group/SMACT.git

# ── Pre-download CHGNet weights at build time ───────────────────────────────
# Avoids runtime download on cold start inside container
RUN python -c "from chgnet.model import CHGNet; CHGNet.load(); print('CHGNet weights cached')"

# ── Copy Chemeleon source code ──────────────────────────────────────────────
COPY chemeleon/ ./chemeleon/
COPY setup.py .
COPY README.md .

# ── Install Chemeleon as a package ──────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Copy model checkpoints (all 4 — ~4.3GB) ────────────────────────────────
# These are baked into the image — no download needed at runtime
COPY chemeleon/checkpoints/ ./chemeleon/checkpoints/

# ── Copy pipeline scripts ───────────────────────────────────────────────────
COPY validator.py .
COPY chgnet_layer.py .

# ── Copy assets and app ─────────────────────────────────────────────────────
COPY assets/ ./assets/
COPY app/ ./app/

# ── Create results directories ──────────────────────────────────────────────
RUN mkdir -p results/prompt \
             results/navigate \
             results/composition \
             results/relaxed

# ── Environment variables ───────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch_cache
# MP_API_KEY must be passed at runtime via --env flag or .env file
# Do NOT hardcode it here

# ── Default command — runs the full pipeline ────────────────────────────────
CMD ["python", "-m", "chemeleon.cli", "--help"]