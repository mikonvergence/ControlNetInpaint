from setuptools import setup

setup(
    name='controlnetinpaint',
    version='0.1',    
    description='ControlNet Inpainting with StableDiffusion',
    url='https://github.com/mikonvergence/ControlNetInpaint',
    author='Mikolaj Czerkawski',
    author_email="mikolaj.czerkawski@esa.int",
    package_dir={"controlnetinpaint":"src"},
    install_requires=[
      "torch>=1.10.0",
      "torchvision",
      "numpy",
      "tqdm",
      "pillow",
      "diffusers==0.14.0",
      "xformers",
      "transformers",
      "scipy",
      "ftfy",
      "accelerate",
      "controlnet_aux"
    ],
)
