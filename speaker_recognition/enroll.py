"""
Builds embeddings.json from reference clips.

Folder layout expected:
  actors/
    Feroze Khan/
      clip1.wav
      clip2.wav
    Sajal Aly/
      clip1.wav
      ...

Use clean, solo-speaker clips, ~15-30s each. Multiple clips per actor
(different scenes/emotions) improve robustness more than one long clip.
"""
import argparse
import json
import os

from .wespeaker_embedder import WeSpeakerEmbedder


def build_embeddings(actors_dir, output_path, device="cuda", model_dir="models/samresnet100_voxblink2"):
    embedder = WeSpeakerEmbedder(model_dir=model_dir, device=device)
    db = {}

    for actor_name in sorted(os.listdir(actors_dir)):
        actor_dir = os.path.join(actors_dir, actor_name)
        if not os.path.isdir(actor_dir):
            continue

        clip_embeddings = []
        for filename in sorted(os.listdir(actor_dir)):
            if not filename.lower().endswith(".wav"):
                continue
            path = os.path.join(actor_dir, filename)
            print(f"Extracting: {actor_name} / {filename}")
            clip_embeddings.append(embedder.extract_from_file(path).tolist())

        if clip_embeddings:
            db[actor_name] = clip_embeddings
            print(f"  -> {actor_name}: {len(clip_embeddings)} clip(s) enrolled")
        else:
            print(f"  WARNING: no .wav clips found for {actor_name}, skipping")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved {len(db)} actors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors-dir", default="actors")
    parser.add_argument("--output", default="embeddings.json")
    parser.add_argument("--model-dir", default="models/samresnet100_voxblink2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    build_embeddings(args.actors_dir, args.output, args.device, args.model_dir)