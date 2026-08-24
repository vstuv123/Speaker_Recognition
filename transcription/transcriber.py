"""
Word-level transcription using faster-whisper.
"""

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="large-v3-turbo", device="cuda", compute_type="float16"):
        print("Loading faster-whisper...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_file,
        language="ur",
        beam_size=5,
        vad_min_silence_ms=300,
        condition_on_previous_text=False,
        initial_prompt=None,
    ):
        print("Transcribing with word timestamps...")

        segments, info = self.model.transcribe(
            audio_file,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=vad_min_silence_ms),
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
        )

        segments = list(segments)

        word_items = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                word_items.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                })

        print(f"Total words with timestamps: {len(word_items)}")
        return word_items, info
