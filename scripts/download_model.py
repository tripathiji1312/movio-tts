"""Download fine-tuned model from wandb artifact."""
import os
import wandb

os.environ["WANDB_API_KEY"] = "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"

api = wandb.Api()
artifact = api.artifact("tripathiji1312-vit/movio-tts/movio-tanglish-f5tts:v2")

dest = "models/indicf5_tanglish_v2"
print(f"Downloading to {dest} ...")
artifact.download(root=dest)
print("Done.")
print(f"Files: {list(__import__('pathlib').Path(dest).iterdir())}")
