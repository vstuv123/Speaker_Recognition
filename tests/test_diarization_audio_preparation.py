import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_processing.preprocessing import prepare_diarization_audio


class DiarizationAudioPreparationTests(unittest.TestCase):
    def test_stereo_audio_is_downmixed_resampled_and_written_as_pcm16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            output = root / "nested" / "diarization.wav"
            sample_rate = 48000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            stereo = np.column_stack((
                0.5 * np.sin(2 * np.pi * 220 * time),
                0.25 * np.sin(2 * np.pi * 440 * time),
            ))
            sf.write(source, stereo, sample_rate, subtype="PCM_24")

            result = prepare_diarization_audio(str(source), str(output))

            self.assertEqual(result, str(output))
            info = sf.info(output)
            self.assertEqual(info.format, "WAV")
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertAlmostEqual(info.duration, 1.0, places=3)

    def test_compatible_input_is_copied_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            output = root / "diarization.wav"
            samples = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
            sf.write(source, samples, 16000, subtype="PCM_16")
            original_bytes = source.read_bytes()

            prepare_diarization_audio(str(source), str(output))

            self.assertEqual(output.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
