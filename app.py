import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from typing import Optional
import base64, io
from pydantic import BaseModel, Field
import inferless
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Add a hat to the cat")
    image_url: Optional[str] = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    guidance_scale: Optional[float] = 2.5
    num_inference_steps: Optional[int] = 30

@inferless.response
class ResponseObjects(BaseModel):
    edited_image_base64: str = Field(default="Test Output")

class InferlessPythonModel:
    def initialize(self):
        model_id = "black-forest-labs/FLUX.1-Kontext-dev"
        self.pipe = FluxKontextPipeline.from_pretrained(model_id,torch_dtype=torch.bfloat16).to("cuda")

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        image = load_image(inputs.image_url)
        out = self.pipe(
            image=image,
            prompt=inputs.prompt,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps,
        ).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return ResponseObjects(edited_image_base64=encoded)

    def finalize(self):
        self.pipe = None
