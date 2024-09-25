

from .func import *
from comfy.utils import ProgressBar

NODE_NAME = 'CatVTON_Wrapper'

class LS_CatVTON:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # device_list = ['cuda', 'cpu', "mps", "xpu"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "refer_image": ("IMAGE",),
                "mask_grow": ("INT", {"default": 25, "min": -999, "max": 999, "step": 1}),
                "mixed_precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 14.0, "step": 0.1, "round": 0.01,},),
                # "device": (device_list,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "catvton"
    CATEGORY = '😺dzNodes/CatVTON Wrapper'

    def catvton(self, image, mask, refer_image, mask_grow, mixed_precision, seed, steps, cfg):

        device = "cuda"
        catvton_path = os.path.join(folder_paths.models_dir, "CatVTON")
        sd15_inpaint_path = os.path.join(catvton_path, "stable-diffusion-inpainting")

        mixed_precision = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]

        pipeline = CatVTONPipeline(
            base_ckpt=sd15_inpaint_path,
            attn_ckpt=catvton_path,
            attn_ckpt_version="mix",
            weight_dtype=mixed_precision,
            use_tf32=True,
            device=device
        )

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = mask[0]
        if mask_grow:
            mask = expand_mask(mask, mask_grow, 0)
        mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        image, refer_image, mask_image = [_.squeeze(0).permute(2, 0, 1) for _ in
                                                 [image, refer_image, mask_image]]
        target_image = to_pil_image(image)
        refer_image = to_pil_image(refer_image)
        mask_image = mask_image[0]
        mask_image = to_pil_image(mask_image)

        generator = torch.Generator(device=device).manual_seed(seed)
        person_image, person_image_bbox = resize_and_padding_image(target_image, (768, 1024))
        cloth_image, _ = resize_and_padding_image(refer_image, (768, 1024))
        mask, _ = resize_and_padding_image(mask_image, (768, 1024))
        mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                           do_convert_grayscale=True)
        mask = mask_processor.blur(mask, blur_factor=9)

        # Inference
        comfyui_pbar_update = ProgressBar(total=steps).update
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            comfy_pbar_callback=comfyui_pbar_update
        )[0]

        result_image = restore_padding_image(result_image, target_image.size, person_image_bbox)
        result_image = to_tensor(result_image).permute(1, 2, 0).unsqueeze(0)
        log(f"{NODE_NAME} Processed.", message_type='finish')

        return (result_image,)

NODE_CLASS_MAPPINGS = {
    "CatVTONWrapper": LS_CatVTON
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CatVTONWrapper": "CatVTON Wrapper"
}