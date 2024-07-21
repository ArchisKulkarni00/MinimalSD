import gc
import os
import warnings

import diffusers
from compel import Compel
import torch
from SD15.utils import load_yaml_file, process_presets, generate_unique_filename, initialize_logging, print_main_menu
from diffusers import LCMScheduler, AutoencoderTiny
from random import randint


class ApplicationBaseClass:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    configuration = None
    inputs = None
    main_pipeline = None
    upscale_pipeline = None
    logger = None
    seed = None
    positive_prompt = None
    negative_prompt = None
    positive_embeds = None
    negative_embeds = None
    is_model_loaded = False
    choice = None
    SCHEDULERS = {
        "unipc": diffusers.schedulers.UniPCMultistepScheduler,
        "euler_a": diffusers.schedulers.EulerAncestralDiscreteScheduler,
        "euler": diffusers.schedulers.EulerDiscreteScheduler,
        "ddim": diffusers.schedulers.DDIMScheduler,
        "ddpm": diffusers.schedulers.DDPMScheduler,
        "deis": diffusers.schedulers.DEISMultistepScheduler,
        "dpm2": diffusers.schedulers.KDPM2DiscreteScheduler,
        "dpm2-a": diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
        "dpm++_2s": diffusers.schedulers.DPMSolverSinglestepScheduler,
        "dpm++_2m": diffusers.schedulers.DPMSolverMultistepScheduler,
        "dpm++_2m_karras": diffusers.schedulers.DPMSolverMultistepScheduler,
        "heun": diffusers.schedulers.HeunDiscreteScheduler,
        "heun_karras": diffusers.schedulers.HeunDiscreteScheduler,
        "lms": diffusers.schedulers.LMSDiscreteScheduler,
        "lms_karras": diffusers.schedulers.LMSDiscreteScheduler,
        "pndm": diffusers.schedulers.PNDMScheduler,
    }

    # 3 Main functions
    # ----------------
    def load_pipeline(self):
        pass

    def generate_image(self):
        pass

    def unload_pipeline(self):
        if self.main_pipeline:
            del self.main_pipeline
        if self.upscale_pipeline:
            del self.upscale_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Pipeline unloaded. Check the gpu space.")

    # Other supporting functions
    # --------------------------

    def run_this_module(self):
        pass

    def initialize_upscaler(self):
        self.upscale_pipeline = upscaler.Upscaler()
        self.upscale_pipeline.logger = self.logger

    def reload_configurations(self):
        self.configuration = load_yaml_file("configuration.yml")
        self.logger.debug("Configurations file reloaded.")

    def reload_inputs(self):
        self.inputs = load_yaml_file("inputs.yml")
        self.logger.debug("Inputs file reloaded.")

    def set_scheduler(self):
        self.main_pipeline.scheduler = (self.SCHEDULERS[self.configuration['scheduler']].
                                        from_config(self.main_pipeline.scheduler.config))
        self.logger.debug("Loaded scheduler {}.".format(self.configuration['scheduler']))

    def set_lcm(self):
        if os.path.exists(os.path.join(self.configuration['loraDir'], "lcm-sd15-lora.safetensors")):
            self.main_pipeline.load_lora_weights(
                os.path.join(self.configuration['loraDir'], "lcm-sd15-lora.safetensors"))
            self.main_pipeline.scheduler = LCMScheduler.from_config(self.main_pipeline.scheduler.config)
            self.logger.debug("Loaded LCM lora.")
        else:
            self.logger.error("LCM Lora not found. Make sure it is placed in the lora folder.")

    def set_detailer(self):
        if os.path.exists(os.path.join(self.configuration['loraDir'], "sd15_add_detail.safetensors")):
            self.main_pipeline.load_lora_weights(
                os.path.join(self.configuration['loraDir'], "sd15_add_detail.safetensors"))
            self.main_pipeline.fuse_lora(lora_scale=1.0)
            self.logger.debug("Loaded detailer lora.")
        else:
            self.logger.error("Detailer Lora not found. Make sure it is placed in the lora folder.")

    def set_tiny_vae(self):
        if not os.path.join(self.configuration["modelsDir"], "TinyVAE"):
            os.mkdir(os.path.join(self.configuration["modelsDir"], "TinyVAE"))
        try:
            self.main_pipeline.vae = AutoencoderTiny.from_pretrained(
                os.path.join(self.configuration["modelsDir"], "TinyVAE"), torch_dtype=torch.float16)
        except:
            self.logger.error("TinyVAE loading failed, check the required files.")

    def generate_seed(self):
        try:
            # generate seed, use -1 for random seed
            # -------------------------------------
            if self.inputs['seed'] == -1:
                some_integer = randint(0, 99999999)
                self.logger.info("Random seed set to: {}".format(str(some_integer)))
                self.seed = torch.Generator("cpu").manual_seed(some_integer)
            else:
                self.seed = torch.Generator("cpu").manual_seed(self.inputs['seed'])
        except:
            self.logger.error('Failed to set the seed.')

    def process_preset(self):
        # if preset value set to "None" skip processing
        if self.inputs['preset'] == "None":
            return
        try:
            preset_pos, preset_neg = process_presets(self.inputs['preset'], self.configuration['presetYMLFile'])
            self.positive_prompt = self.inputs['positivePrompt'] + preset_pos
            self.negative_prompt = self.inputs['negativePrompt'] + preset_neg
            self.logger.info("Preset set to {}.".format(self.inputs['preset']))
        except:
            self.logger.error("Preset setting failed.")
            self.positive_prompt = self.inputs['positivePrompt']
            self.negative_prompt = self.inputs['negativePrompt']

    def process_prompt_weight(self):
        with torch.no_grad():
            compel = Compel(tokenizer=self.main_pipeline.tokenizer, text_encoder=self.main_pipeline.text_encoder)
            self.positive_embeds = compel([self.positive_prompt])
            # self.negative_embeds = compel([self.negative_prompt])

    def save_image(self, image, index):
        if not os.path.exists(self.configuration['outputDir']):
            os.mkdir(self.configuration['outputDir'])

        image.save(
            os.path.join(self.configuration['outputDir'],
                         generate_unique_filename(self.configuration['imageNamePrefix'], index)))
        if self.configuration["isPreviewEnabled"] == "yes":
            image.show()
        self.logger.info("Saving image {} of {}.".format(index + 1, self.inputs['numOfImages']))
