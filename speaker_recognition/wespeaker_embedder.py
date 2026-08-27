"""
WeSpeaker backend using the stable CLI interface (subprocess) instead of
the internal Python API, which has changed across wespeaker versions.
CUDA is broken in this wespeaker CLI build (feature tensor never moved to
GPU), so this backend always runs on CPU -- fine for short enrollment/
identification clips.
"""
import os
import subprocess
import tempfile

import numpy as np


class WeSpeakerEmbedder:
    def __init__(self, model_dir, device="cpu"):  # device arg kept for interface parity
        self.model_dir = model_dir

    def _run_cli(self, audio_path):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        cmd = [
            "wespeaker",
            "--task", "embedding",
            "--audio_file", audio_path,
            "--pretrain", self.model_dir,
            "--device", "cpu",
            "--output_file", output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            embedding = np.loadtxt(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
        return embedding

    def extract_from_file(self, path):
        return self._run_cli(path)

    def extract(self, waveform, sample_rate):
        """Same interface as EcapaEmbedder -- writes a temp WAV since the
        CLI needs a file path, not a raw array."""
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, waveform, sample_rate)
            return self._run_cli(tmp_path)
        finally:
            os.remove(tmp_path)