from setuptools import find_packages
from distutils.core import setup

setup(
    name='humanoidGym',
    version='1.0.0',
    author='Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='rudinn@ethz.ch',
    description='Isaac Gym environments for HumanoidGym-Yuhan',
    install_requires=['isaacgym',
                    #   'rsl-rl',
                      'matplotlib', 
                      'numpy==1.21.6', 
                      'tensorboard', 
                      'mujoco==3.2.3', 
					  'mujoco-python-viewer',
                      'wandb',
                      'tqdm>=4.0',
                      'opencv-python',
                      'pyyaml',
                      'onnx',
                      'coloredlogs',
                      "torch>=1.4.0",
                      "torchvision>=0.5.0",
                      ]
)