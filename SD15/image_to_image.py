import os

import diffusers
import torch
from SD15 import upscaler
from SD15.ApplicationBaseClass import ApplicationBaseClass
from SD15.utils import print_main_menu, initialize_logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


class ImageToImage(ApplicationBaseClass):
    input_image = None

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

            if self.configuration['isUpscalingEnabled'] == 'yes':
                self.initialize_upscaler()
                self.upscale_pipeline.main_pipeline = StableDiffusionImg2ImgPipeline(**self.main_pipeline.components)

            # add lcm and detailer loras
            if self.configuration['isLCMEnabled'] == "yes":
                self.set_lcm()

            if self.configuration['isDetailerEnabled'] == "yes":
                self.set_detailer()

            # move the model to cuda device
            if torch.cuda.is_available():
                self.main_pipeline.to("cuda")
                self.logger.debug("Model loaded on CUDA device.")

            self.logger.info("Model loaded successfully.")
        else:
            self.logger.error("Model not found. Please check the configurations.")



    # runs the inference on the pipeline to generate the images using input data, and saves them
    # ------------------------------------------------------------------------------------------
    def generate_image(self):
        self.reload_inputs()
        # if the code is called externally by t2i, then we set all the parameters there, else we set here
        self.generate_seed()
        self.process_preset()
        self.process_prompt_weight()
        self.load_image()

        if self.input_image and self.main_pipeline:
            self.resize_image()
            set_of_images = self.main_pipeline(
                prompt_embeds=self.positive_embeds,
                num_inference_steps=self.inputs['numOfSteps'],
                negative_prompt=self.negative_prompt,
                guidance_scale=self.inputs['guidanceScale'],
                generator=self.seed,
                image=self.input_image,
                strength=self.inputs['i2iStrength']
            ).images

            self.input_image = None

            # if upscaler is enabled the final output is saved by the upscaler, else it is saved by this file
            for index in range(self.inputs['numOfImages']):
                low_res_image = set_of_images[index]
                if self.configuration['isUpscalingEnabled'] == 'yes':
                    self.upscale_fix(low_res_image)
                else:
                    self.save_image(low_res_image, index)

    # load input image
    def load_image(self):
        try:
            self.input_image = Image.open(self.inputs['inputImage'])
            self.logger.info("Input image set.")
        except:
            self.logger.error('Loading input image failed.')

    # resize the image to max of 768 since generated image size is same as input image size
    def resize_image(self):
        input_image_width, input_image_height = self.input_image.size
        if input_image_width <= 768 and input_image_height <= 768:
            return
        if input_image_width > input_image_height:
            new_width = 768
            new_height = int((768 * input_image_height) / input_image_width)
            # diffusers accepts sizes divisible by 8. hence we calculate the next closest dimension divisible by 8
            new_height = new_height + (8 - (new_height % 8))
        else:
            new_height = 768
            new_width = int((768 * input_image_width) / input_image_height)
            # diffusers accepts sizes divisible by 8. hence we calculate the next closest dimension divisible by 8
            new_width = new_width + (8 - (new_width % 8))
        self.input_image = self.input_image.resize((new_width, new_height))

    # initialize data in upscaler and trigger image build
    def upscale_fix(self, low_res_image):
        self.upscale_pipeline.positive_embeds = self.positive_embeds
        self.upscale_pipeline.negative_prompt = self.negative_prompt
        self.upscale_pipeline.seed = self.seed
        self.upscale_pipeline.reload_configurations()
        self.upscale_pipeline.low_res_image = low_res_image
        self.upscale_pipeline.generate_image()

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
                self.logger.warning("Please add a input")

            # load the pipeline
            if self.choice == 1:
                if not self.is_model_loaded:
                    self.load_pipeline()
                    if self.main_pipeline:
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
    text_to_image = ImageToImage()
    text_to_image.run_this_module()
