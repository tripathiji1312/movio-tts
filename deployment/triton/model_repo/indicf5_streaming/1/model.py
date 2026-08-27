import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from transformers import AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_id = os.environ.get("MODEL_ID", "ai4bharat/IndicF5")
        self.num_steps = int(os.environ.get("NUM_FLOW_STEPS", "12"))
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        voices_path = os.path.join(args["model_repository"], args["model_version"], "voices.json")
        if os.path.exists(voices_path):
            with open(voices_path) as fh:
                self.voices = json.load(fh)
        else:
            self.voices = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = text_tensor.as_numpy()[0][0].decode("utf-8")
            voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE")

            payload = {"text": text}
            if voice_tensor is not None:
                voice = voice_tensor.as_numpy()[0][0].decode("utf-8")
                ref = self.voices.get(voice)
                if ref:
                    payload["ref_audio"] = ref["ref_audio"]
                    payload["ref_text"] = ref["ref_text"]

            audio = np.asarray(self.model(payload), dtype=np.int16)
            out = pb_utils.Tensor("PCM_CHUNK", audio.reshape(1, -1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        del self.model
