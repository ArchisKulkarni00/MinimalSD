import os

import diffusers
import torch
from SD15 import upscaler
from SD15.ApplicationBaseClass import ApplicationBaseClass
from SD15.utils import print_main_menu, initialize_logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


class TextToImage(ApplicationBaseClass):

    # Loads the stable diffusion pipeline, assigns scheduler, adds loras, moves to gpu
    # --------------------------------------------------------------------------------
    def load_pipeline(self):
        self.reload_configurations()
        model_path = os.path.join(self.configuration['modelsDir'], self.configuration['modelName'])
        if os.path.exists(model_path):
            self.main_pipeline = StableDiffusionPipeline.from_single_file(
                pretrained_model_link_or_path=model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True)
            self.logger.debug("Model found. Pipeline created")
        else:
            self.logger.error("Model not found. Please check the configurations.")

        self.set_scheduler()

        # add lcm and detailer loras
        if self.configuration['isLCMEnabled'] == "yes":
            if os.path.exists(os.path.join(self.configuration['loraDir'], "lcm-sd15-lora.safetensors")):
                self.set_lcm()
            else:
                self.logger.error("LCM Lora not found. Make sure it is placed in the lora folder.")

        if self.configuration['isDetailerEnabled'] == "yes":
            self.set_detailer()
        else:
            self.logger.error("Detailer Lora not found. Make sure it is placed in the lora folder.")

        # move the model to cuda device
        if torch.cuda.is_available():
            self.main_pipeline.to("cuda")
            self.logger.debug("Model loaded on CUDA device.")

        self.logger.info("Model loaded successfully.")

    # runs the inference on the pipeline to generate the images using input data, and saves them
    # ------------------------------------------------------------------------------------------
    def generate_image(self):
        self.reload_inputs()
        self.generate_seed()
        self.process_preset()

        if self.main_pipeline:
            set_of_images = self.main_pipeline(
                self.positive_prompt,
                num_inference_steps=self.inputs['numOfSteps'],
                negative_prompt=self.negative_prompt,
                height=self.inputs['heightOfImage'],
                width=self.inputs['widthOfImage'],
                guidance_scale=self.inputs['guidanceScale'],
                generator=self.seed,
                num_images_per_prompt=self.inputs['numOfImages']
            ).images

            # if upscaler is enabled the final output is saved by the upscaler, else it is saved by this file
            for index in range(self.inputs['numOfImages']):
                low_res_image = set_of_images[index]
                if self.configuration['isUpscalingEnabled'] == 'yes':
                    self.upscale_fix(low_res_image)
                else:
                    self.save_image(low_res_image, index)

    def upscale_fix(self, low_res_image):
        self.upscale_pipeline = upscaler.Upscaler()
        self.upscale_pipeline.logger = self.logger
        self.upscale_pipeline.positive_prompt = self.positive_prompt
        self.upscale_pipeline.negative_prompt = self.negative_prompt
        self.upscale_pipeline.seed = self.seed
        self.upscale_pipeline.reload_configurations()
        self.upscale_pipeline.main_pipeline = StableDiffusionImg2ImgPipeline(**self.main_pipeline.components)
        self.upscale_pipeline.low_res_image = low_res_image
        self.upscale_pipeline.generate_image()

    # runner function that handles main menu, user input etc. used when running this file independently
    # -------------------------------------------------------------------------------------------------
    def run_this_module(self):
        diffusers.utils.logging.set_verbosity(50)
        self.logger = initialize_logging()
        self.is_model_loaded = False
        while True:
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
    text_to_image = TextToImage()
    text_to_image.run_this_module()
