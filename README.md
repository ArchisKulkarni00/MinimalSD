# MinimalSD
Minimalistic stable diffusion based on 🤗 Diffusers library.

## Overview
### What this application is?
This is a minimal implementation of the stable diffusion pipeline for various tasks.

It is expected to be minimal on the system, simple to use and the code should be easier to understand.

### What this application is not?

An alternative to existing applications like A1111, ComfyUI, Fooocus, Invoke or SD webui. These are already matured pieces of software and I do not intend to start a race with them.

### Then why this application?

Simply put i wanted to do some shenanigans with stable diffusion using pure code rather than a UI. Hence i created this application which is quite close to pure code and also has a minimal interface, thus serving the needs of programmers and non-programming folks alike. 

This  application can be used as a standalone app or as a base library for creating stable diffusion based applications.

## Installation

Currently I have not compiled the code to any distributable format like exe etc, and that is planned for future. Right now it expects python to be installed on the system to run. Follow the instructions below:
1. Clone the repo, or download the source code.
2. Create a virtual env and activate it.(optional step)
3. Open terminal or powershell in the folder with requirements.txt and type "python requirements.txt".
4. It should install most of the required packages.
5. Pytorch needs to be installed separately as it can be platform dependent for CUDA enabled cards. Head to [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) 
6. Select appropriate OS, stable build, package, language and platform. Copy the generated command and paste in same terminal/powershell.

### Running the code
Open terminal and head to SD15 folder and run:
```commandline
python text_to_image_.py
```
*Replace text_to_image_.py with image_to_image_.py to run image to image pipeline*