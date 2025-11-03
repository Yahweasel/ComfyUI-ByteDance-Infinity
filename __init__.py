import os
import os.path as osp
import torch
from torch.cuda.amp import autocast
import folder_paths
import comfy.model_management as mm
from .tools import run_infinity
from .infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates



INFINITY_MODELS_DIR = os.path.join(folder_paths.models_dir, "infinity", "model")
INFINITY_VAES_DIR = os.path.join(folder_paths.models_dir, "infinity", "vae")
INFINITY_TEXT_ENCODERS_DIR = os.path.join(folder_paths.models_dir, "infinity", "clip")

folder_paths.add_model_folder_path("infinity_model", INFINITY_MODELS_DIR)
folder_paths.add_model_folder_path("infinity_vae", INFINITY_VAES_DIR)
folder_paths.add_model_folder_path("infinity_text_encoder", INFINITY_TEXT_ENCODERS_DIR)

class InfinityTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("STRING", {"default": "flan-t5-xl"}),
                "device": (["default", "cpu"],),
            }
        }

    RETURN_TYPES = ("INFINITY_TEXT_ENCODER",)
    FUNCTION = "load"

    def load(self, text_encoder, device):
        if device == "default":
            device = str(mm.get_torch_device())
        text_encoder_ckpt = osp.join(INFINITY_TEXT_ENCODERS_DIR, text_encoder)
        text_tokenizer, text_encoder = run_infinity.load_tokenizer(
            device,
            t5_path=text_encoder_ckpt
        )
        return ((text_tokenizer, text_encoder),)

class InfinityVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_type": ([14, 16, 18, 20, 24, 32, 64], {"default": 32}),
                "vae": (folder_paths.get_filename_list("infinity_vae"),),
                "device": (["default", "cpu"],),
                "apply_spatial_patchify": ([False, True],),
            }
        }

    RETURN_TYPES = ("INFINITY_VAE",)
    FUNCTION = "load"

    def load(self, vae_type, vae, device, apply_spatial_patchify):
        if device == "default":
            device = str(mm.get_torch_device())
        class Args:
            pass
        args = Args()
        args.vae_type = vae_type
        args.vae_path = osp.join(INFINITY_VAES_DIR, vae)
        args.apply_spatial_patchify = apply_spatial_patchify
        vae = run_infinity.load_visual_tokenizer(device, args)
        return ((vae, args),)

class InfinityModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("INFINITY_VAE",),
                "model": ("STRING", {"default": "infinity_2b_reg.pth"}),
                "checkpoint_type": (["torch", "torch_shard"],),

                "pn": (['0.06M', '0.25M', '1M'], {"default": '1M'}),
                "model_type": ("STRING", {"default": "infinity_2b"}),
                "rope2d_each_sa_layer": ([False, True], {"default": True}),
                "rope2d_normalized_by_hw": ([0, 1, 2], {"default": 2}),
                "use_scale_schedule_embedding": ([False, True],),
                "use_bit_label": ([False, True], {"default": True}),
                "add_lvl_embeding_only_first_block": ([False, True],),
                "text_channels": ("INT", {"default": 2048}),
                "use_flex_attn": ([False, True],),
                "bf16": ([False, True], {"default": True}),
            }
        }

    RETURN_TYPES = ("INFINITY_MODEL",)
    FUNCTION = "load"

    def load(
        self,
        vae,
        model,
        checkpoint_type,
        pn,
        model_type,
        rope2d_each_sa_layer,
        rope2d_normalized_by_hw,
        use_scale_schedule_embedding,
        use_bit_label,
        add_lvl_embeding_only_first_block,
        text_channels,
        use_flex_attn,
        bf16
    ):
        vae, vae_args = vae

        class Args:
            pass
        args = Args()
        args.model_path = osp.join(INFINITY_MODELS_DIR, model)
        args.pn = pn
        args.model_type = model_type
        args.rope2d_each_sa_layer = int(rope2d_each_sa_layer)
        args.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        args.use_scale_schedule_embedding = int(use_scale_schedule_embedding)
        args.use_bit_label = int(use_bit_label)
        args.add_lvl_embeding_only_first_block = int(add_lvl_embeding_only_first_block)
        args.text_channels = text_channels
        args.use_flex_attn = int(use_flex_attn)
        args.bf16 = bf16

        args.apply_spatial_patchify = vae_args.apply_spatial_patchify
        args.checkpoint_type = checkpoint_type
        args.cache_dir = 'DISABLE-MODEL-CACHE'
        args.enable_model_cache = 0

        infinity = run_infinity.load_transformer(mm.get_torch_device(), vae, args)
        return ((infinity, args),)

class Infinity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INFINITY_MODEL",),
                "vae": ("INFINITY_VAE",),
                "text_encoder": ("INFINITY_TEXT_ENCODER",),

                "prompt": ("STRING", {"default": "a dog", "multiline": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("STRING", {"default": "3"}),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0}),
                "cfg_insertion_layer": ("INT", {"default": 0}),
                "sampling_per_bits": ([1, 2, 4, 8, 16],),
                "h_div_w_template": (tuple(dynamic_resolution_h_w.keys()),),
                "enable_positive_prompt": ([False, True],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    def generate(
        self,
        model,
        vae,
        text_encoder,
        prompt,
        seed,
        cfg,
        tau,
        cfg_insertion_layer,
        sampling_per_bits,
        h_div_w_template,
        enable_positive_prompt
    ):
        class Args:
            pass
        args = Args()
        args.cfg = cfg
        args.tau = tau
        args.cfg_insertion_layer = cfg_insertion_layer
        args.sampling_per_bits = sampling_per_bits
        args.h_div_w_template = h_div_w_template
        args.enable_positive_prompt = enable_positive_prompt
        args.seed = seed
        args.prompt = prompt

        # parse cfg
        args.cfg = list(map(float, args.cfg.split(',')))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
        
        # load text encoder
        text_tokenizer, text_encoder = text_encoder
        # load vae
        vae, vae_args = vae
        args.vae_type = vae_args.vae_type
        args.apply_spatial_patchify = vae_args.apply_spatial_patchify
        # load infinity
        infinity, model_args = model
        args.pn = model_args.pn
        
        scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
        scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                generated_image = run_infinity.gen_one_img(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    args.prompt,
                    g_seed=args.seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=args.enable_positive_prompt,
                )
        return (generated_image[:, :, [2, 1, 0]].unsqueeze(0).float() / 255.0,)

NODE_CLASS_MAPPINGS = {
    "Infinity: Load model": InfinityModel,
    "Infinity: Load VAE": InfinityVAE,
    "Infinity: Load text encoder": InfinityTextEncoder,
    "Infinity: Generate image": Infinity,
}
