import os
import sys
import traceback

import huggingface_hub
import pyannote.audio
from pyannote.audio import Pipeline

import os

FFMPEG_BIN = r"E:\ffmpeg7-shared\bin"

os.add_dll_directory(FFMPEG_BIN)
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]

import torch
from pyannote.audio import Pipeline

print(f"huggingface_hub version: {huggingface_hub.__version__}")
print(f"pyannote.audio version: {pyannote.audio.__version__}")
print(f"python version: {sys.version}")

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN environment variable is not set.")
    sys.exit(1)

print("\nAttempting to load pyannote/speaker-diarization-community-1...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    print("\nSUCCESS: pipeline loaded correctly.")
except Exception:
    print("\nFAILED:")
    traceback.print_exc()