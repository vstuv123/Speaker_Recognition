"""
Speaker embedding extraction. ECAPA-TDNN (SpeechBrain) is the baseline
backend -- other backends (WeSpeaker SimAM-ResNet100, ResNet293, etc.)
should implement the same extract()/extract_from_file() interface so
identify.py doesn't need to change when swapping models.
"""
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class EcapaEmbedder:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def extract(self, waveform, sample_rate):
        """waveform: 1D numpy float32 array (mono). Returns 192-d numpy embedding."""
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sample_rate, 16000
            ).numpy()
        signal = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_batch(signal)
        return emb.squeeze().cpu().numpy()

    def extract_from_file(self, path):
        signal, sr = torchaudio.load(path)
        signal = signal.mean(dim=0)  # downmix to mono
        return self.extract(signal.numpy(), sr)