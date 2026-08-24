"""
Vocal separation using Demucs.

Splits a mixed audio file (dialogue + music + effects) into a
"vocals" stem and everything else, so that diarization and
transcription only have to deal with speech.
"""

import os
import subprocess


def separate_vocals(input_path, output_dir, model="htdemucs", device="cuda"):
    """
    Runs Demucs to separate vocals/dialogue from the rest of the mix.

    Returns the path to the extracted vocals.wav file.
    """

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "-d", device,
        "-o", output_dir,
        input_path,
    ]

    print(f"Running Demucs vocal separation: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Demucs CLI not found. Install it with: pip install demucs"
        ) from exc
    except subprocess.CalledProcessError as exc:
        if device == "cuda":
            print("Demucs failed on GPU, retrying on CPU...")
            cmd[cmd.index("-d") + 1] = "cpu"
            subprocess.run(cmd, check=True)
        else:
            raise exc

    track_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, model, track_name, "vocals.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(
            f"Demucs did not produce the expected output at {vocals_path}"
        )

    print(f"Vocal separation complete: {vocals_path}")
    return vocals_path
