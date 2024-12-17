# MinimalSD 🖌️

Minimalistic Stable Diffusion, powered by 🤗 Diffusers Library

---

## Overview

### What _is_ MinimalSD?

MinimalSD is a lightweight, code-focused implementation of the Stable Diffusion pipeline, designed for various creative tasks. It's engineered to keep things easy on system resources, simple to use, and straightforward for developers to understand.

### What MinimalSD _isn't_!

This isn't a replacement for robust, feature-packed apps like A1111, ComfyUI, Fooocus, Invoke, or SD WebUI. Those are powerful, full-fledged programs. MinimalSD is here for those who prefer to get their hands into the code without all the extra frills.

### Why MinimalSD?

Because sometimes, you just want a bit of coding freedom without wrestling with UIs. MinimalSD is built with simplicity in mind, delivering a lean, almost pure-code experience. Whether you're a programmer or just someone who prefers minimalism, this library can serve as a standalone app or as a foundational library for your own Stable Diffusion-based creations. 🎉

---

## Installation 🚀

> **Note**: Currently, there's no executable distribution (like .exe files). MinimalSD runs on Python, so you'll need that installed. But stay tuned—plans for compiled versions are in the works!

1. **Clone** the repo or download the source.
2. **Set up a virtual environment** (optional but recommended):
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    
3. **Install dependencies**:
    
    ```bash
    python -m pip install -r requirements.txt
    ```
    
4. **Install PyTorch**: Go to [PyTorch Start Locally](https://pytorch.org/get-started/locally/) to download the version compatible with your system, especially if you’re using CUDA-enabled GPUs.

---

### Optional: Installing Tiny VAE

Tiny VAE is a streamlined version of the Variational Autoencoder used in Stable Diffusion, cutting down generation time and memory load—at a slight cost to image quality. Here’s how to add it:

1. Visit [Tiny VAE Files](https://huggingface.co/madebyollin/taesd/tree/main).
2. Download `diffusion_pytorch_model.safetensors` and `config.json`.
3. Place these files in your models directory under `your_models_folder/TinyVAE/`.

---

## Running the Code 🏃

In your terminal, navigate to the SD15 folder and run:

```bash
python text_to_image_.py
```

Switch to `image_to_image_.py` to use the image-to-image pipeline.

### File and directory structure
Under SD15 folder you will find:
- **/input-images**: stores external images useful as inputs, for image to image or upscaling 
- **/jupyter-notebooks**: minimal jupyter notebook version of the codes
- **/loras**: stores all the lora files
- **/models**: stores all the model files
- **/outputs**: stores output of images
- **configuration.yml**: stores all the configurations for stable diffusion. think like settings for the applications. read every time the mode is loaded to the memory. if some changes are made to this, reload the model to apply the changes.
- **inputs.yml**: stores the input information for the model, including the prompts, guidance scale, image paths etc. this is where you prompt, save and run inference. read every time image is generated.
- **text_to_image.py**: contains code for text to image
- **image_to_image.py**: contains code for image to image

### Basic usage flow
```
run text_to_image.py --> set your settings in configurations.yml --> load the model --> update the prompt and other fields in inputs.yml --> Generate images --> keep updating prompt and generating images --> remove the model
```
---
## Screenshots

Coming soon...😉

---
## Features 🎨

**Implemented so far:**

- ✔️ Text-to-Image (SD 1.5)
- ✔️ Image-to-Image (SD 1.5)
- ✔️ Upscaling (SD 1.5)
- ✔️ Presets
- ✔️ Prompt weighting (see comple library for more details)
- ✔️ Image previews
- ✔️ Tiny VAE support
- ✔️ LoRA (Low-Rank Adaptation) integration
- ✔️ Metadata saving within images
- ✔️ Commented configurations and YAML input files

**On the To-Do List:**

- 🔲 Input validation and configuration checks
- 🔲 Bulk input option via CSV
- 🔲 Testing suite
- 🔲 API support
- 🔲 Automated model downloading and setup
- 🔲 Report generation
- 🔲 List of checkpoints, LoRAs, and components for download
- 🔲 Multi-image generation at the upscaler level
- 🔲 Distributable setup
- 🔲 Example code snippets
- 🔲 Detailed usage and specifications guide

---

### Get in Touch!

If you have ideas, improvements, or just want to geek out about diffusion models, feel free to open an issue or submit a pull request!

