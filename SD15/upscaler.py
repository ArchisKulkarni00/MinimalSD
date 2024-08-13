import os

import diffusers
import torch
from PIL import ImageEnhance
from PIL import Image
from SD15.ApplicationBaseClass import ApplicationBaseClass
from SD15.utils import print_main_menu, initialize_logging
from diffusers import StableDiffusionImg2ImgPipeline


class Upscaler(ApplicationBaseClass):
    low_res_image = None
    steps_multiplier = 1.0

    # Loads the stable diffusion pipeline, assigns scheduler, adds loras, moves to gpu
    # --------------------------------------------------------------------------------
    def load_pipeline(self):
        self.reload_configurations()
        model_path = os.path.join(self.configuration['modelsDir'], self.configuration['modelName'])
        if os.path.exists(model_path):
            self.main_pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                pretrained_model_link_or_path=model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True)
            self.logger.debug("Model found. Pipeline created")

            # set scheduler, fast (tiny) vae, initialize upscaler
            self.set_scheduler()

            if self.configuration['isFastVAEEnabled'] == "yes":
                self.set_tiny_vae()

            # add lcm and detailer loras
            if self.configuration['isLCMEnabled'] == "yes":
                self.set_lcm()

            if self.configuration['isDetailerEnabled'] == "yes":
                self.set_detailer()

            # move the model to cuda device
            if torch.cuda.is_available():
                self.main_pipeline.to("cuda")
                self.logger.debug("Upscale model loaded on CUDA device.")

            self.logger.info("Upscale model loaded successfully.")
        else:
            self.logger.error("Model not found. Please check the configurations.")

    # runs the inference on the pipeline to generate the images using input data, and saves them
    # ------------------------------------------------------------------------------------------
    def generate_image(self):
        # reload inputs, generate seed, process the preset, process prompt weights
        self.reload_inputs()
        self.guidance_scale = float(self.inputs['guidanceScale'])
        # if the code is called externally by t2i, then we set all the parameters there, else we set here
        if not self.low_res_image:
            self.load_image()
            self.steps_multiplier = 1.5
            image_metadata = self.low_res_image.text
            if "positive_prompt" in image_metadata:
                self.positive_prompt = image_metadata['positive_prompt']
                self.negative_prompt = image_metadata['negative_prompt']
                self.seed_int = int(image_metadata['seed'])
                self.seed = torch.Generator("cpu").manual_seed(self.seed_int)
                self.process_prompt_weight()
            else:
                self.process_preset()
                self.logger.debug("No metadata found.")

            self.no_of_steps = self.inputs['numOfSteps']

        upscaler_input_image = self.cheap_upscale()

        if self.main_pipeline:
            set_of_images = self.main_pipeline(
                prompt_embeds=self.positive_embeds,
                num_inference_steps=int(self.no_of_steps * self.steps_multiplier),
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                generator=self.seed,
                image=upscaler_input_image,
                strength=self.inputs['upscalerStrength']
            ).images

            self.low_res_image = None

            for index in range(self.inputs['numOfImages']):
                image = set_of_images[index]
                self.save_image(image, index)

    def load_image(self):
        try:
            self.low_res_image = Image.open(self.inputs['upscaleInputImage'])
            self.logger.info("Input image set.")
        except:
            self.logger.error('Loading input image failed.')

    def cheap_upscale(self):
        temp_image = (self.low_res_image.resize
                      ((self.low_res_image.width * 2, self.low_res_image.height * 2), resample=Image.LANCZOS))
        temp_image = ImageEnhance.Sharpness(temp_image).enhance(3.0)
        return temp_image

    # runner function that handles main menu, user input etc. used when running this file independently
    # -------------------------------------------------------------------------------------------------
    def run_this_module(self):
        diffusers.utils.logging.set_verbosity(50)
        self.logger = initialize_logging()
        self.is_model_loaded = False

        # run a infinite loop inputing user choice of operation
        while True:
            self.choice = None
            print_main_menu()
            try:
                self.choice = int(input('Enter your choice:'))
            except ValueError:
                self.logger.warn("Please add a input")

            # load the pipeline
            if self.choice == 1:
                if not self.is_model_loaded:
                    self.load_pipeline()
                    self.is_model_loaded = True
                else:
                    self.logger.info('Model already loaded, please unload it first.')

            # generate the image if model is loaded
            elif self.choice == 2:
                if not self.is_model_loaded:
                    self.logger.info("Load the model first.")
                else:
                    self.generate_image()

            # Unload the model from memory
            elif self.choice == 3:
                self.unload_pipeline()
                self.is_model_loaded = False
                self.logger.debug("Cache emptied.")
                self.logger.info('Model unloaded, you can check the memory.')

            # get the gpu usage
            elif self.choice == 4:
                usage = torch.cuda.mem_get_info()
                self.logger.info("Memory usage on your CUDA device")
                self.logger.info("--------------------------------")
                self.logger.info("Current Device: " + torch.cuda.get_device_name())
                self.logger.info("Free memory: {} GB".format(str(usage[0] / 1024 / 1024 / 1024)))
                self.logger.info("Used memory: {} GB".format(str((usage[1] - usage[0]) / 1024 / 1024 / 1024)))
                self.logger.info("Total memory: {} GB".format(str(usage[1] / 1024 / 1024 / 1024)))

            # unload the model and exit program
            elif self.choice == 5:
                try:
                    self.unload_pipeline()
                except NameError:
                    self.logger.info("Model already unloaded.")
                self.logger.info("Exiting program. Have a good day!")
                break


if __name__ == "__main__":
    text_to_image = Upscaler()
    text_to_image.run_this_module()
