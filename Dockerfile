# RAVANA cognitive architecture — runtime container
# Root access inside the container (founder-requested) so the set-and-forget loop
# can reset/fix the container if RAVANA breaks. The HOST binds are restricted:
#   - /work is the ONLY writable mount (loop-assigned repo); read-write there
#   - weights/ is mounted read-write so the canonical self persists across runs
#   - no credential files are mounted; RAVANA never sees *.env / *.pem / *.key
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAVANA_OFFLINE=0 \
    RAVANA_WORK_VOLUME=/work

# System deps for numpy/scipy build + git (for the github_cli "hands")
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/ravana

# Install python deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the project source
COPY . .

# Work volume for loop-assigned repos (the only writable host mount)
RUN mkdir -p /work /app/ravana/weights /app/ravana/scratch
VOLUME ["/work", "/app/ravana/weights"]

# Run as root (founder-requested: loop can reset/fix container).
# The hard guards in ravana/agent/tool_registry.py still block destructive ops
# regardless of root — defense in depth.
USER root
EXPOSE 4001
CMD ["python", "scripts/ravana_chat.py"]
