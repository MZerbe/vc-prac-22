# 02- StyleGAN Installation
Repo: https://github.com/NVlabs/stylegan3  
You can find more detailed instructions in the repository.

## Requirements
- CUDA-Installation 11.1 or higher (not the one from conda)
- 64-bit Python 3.8 or higher
- Python libraries (see environment.yml). Installation can be done with conda.
- GCC 9 or later (Linux) or Visual Studio Compilers. If running on Windows be sure to install the "Desktop Build Tools"
  and add the Path ```"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"```
  into your ```PATH``` environment variable.

## Common Mistakes
- Pytorch, torchaudio and torchvision need to be installed as gpu-mode. If this is not the case you need to uninstall
  these and reinstall with the ```-c nvidia``` tag.
- The ```CUDA_HOME``` environment variable is not set correctly towards a valid CUDA installation. This needs to be also
  set in the conda environment ``` conda env config vars set CUDA_HOME=PATHTOCUDA```
- The symlinks of the CUDA-Compiler are not pointing towards valid g++ or gcc compiler. These should be atleast gcc-9 & g++-9.
  Install the correct compiler and set the
- The conda environment does not find the Visual Studio compilers. To fix this run
  ```"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat" ``` in your
  conda environment.
