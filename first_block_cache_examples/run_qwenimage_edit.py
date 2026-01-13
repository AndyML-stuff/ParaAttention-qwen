import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

# Load a sample image (you would replace this with your actual image)
# For demonstration, we'll create a simple test image
image = Image.new("RGB", (512, 512), color=(255, 0, 0))  # Red square

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

prompt = "Transform this red square into a beautiful sunset landscape with mountains and a lake"

# Generate with different aspect ratios
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipe(**inputs)
    output_image = output.images[0]
    output_image.save("qwenimage_edit_fbc_example.png")
    print("Saved edited image to qwenimage_edit_fbc_example.png")
