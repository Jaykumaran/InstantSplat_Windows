
<h2 align="center"> <a href="https://arxiv.org/abs/2403.20309">InstantSplat: Sparse-view SfM-free <a href="https://arxiv.org/abs/2403.20309"> Gaussian Splatting in Seconds </a>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.20309) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/InstantSplat) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://instantsplat.github.io/) [![X](https://img.shields.io/badge/-Twitter@Zhiwen%20Fan%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/WayneINR/status/1774625288434995219)  [![youtube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://youtu.be/fxf_ypd7eD8) [![youtube](https://img.shields.io/badge/Tutorial_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=JdfrG89iPOA&t=347s)
</h5>

<div align="center">
This repository is a modified implementation of InstantSplat, an sparse-view, SfM-free framework for large-scale scene reconstruction method using Gaussian Splatting.
InstantSplat supports 3D-GS, 2D-GS, and Mip-Splatting.
This version works for Windows! Enjoy!
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Free-view Rendering](#free-view-rendering)
- [TODO List](#todo-list)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Free-view Rendering
https://github.com/zhiwenfan/zhiwenfan.github.io/assets/34684115/748ae0de-8186-477a-bab3-3bed80362ad7

## TODO List
- [x] Support 2D-GS
- [ ] Long sequence cross window alignment
- [ ] Support Mip-Splatting

## Get Started

### Installation
1. Clone InstantSplat and download pre-trained model.
```bash
git clone --recursive https://github.com/NVlabs/InstantSplat.git
cd InstantSplat
if not exist "mast3r\checkpoints" mkdir "mast3r\checkpoints"
curl -o mast3r\checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth ^
     https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

2. Create the environment (or use pre-built docker), here we show an example using conda.
```bash
conda create -n instantsplat python=3.10.13 cmake=3.14.0 -y
conda activate instantsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim
pip install plyfile
pip install open3d
pip install "imageio[ffmpeg]"
```

3. **Optional but highly suggested**, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../..
```

4. Download the [run_infer.py](https://github.com/jonstephens85/InstantSplat_Windows/blob/main/run_infer.py) and [instantsplat_gradio.py](https://github.com/jonstephens85/InstantSplat_Windows/blob/main/instantsplat_gradio.py) and place them in the root folder `C:/user/<username>/InstantSplat`

**TROUBLESHOOTING**
If you have CUDA Toolkit 12.6, I ran into issues running: 
```bash
conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
```
I downloaded and installed CUDA Toolkit 11.8. Then set CUDA Toolkit 11.8 for your command session using:
```bash
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_HOME%\bin;%PATH%
set LD_LIBRARY_PATH=%CUDA_HOME%\lib64;%LD_LIBRARY_PATH%
```
Then running 
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can check what version of CUDA Toolkit you are running by using `nvcc --version`
<br><br>

### Data Prep
The original project provides a few examples to try, you can also download their pre-processed data: [link](https://drive.google.com/file/d/1Z17tIgufz7-eZ-W0md_jUlxq89CD1e5s/view)

Place 3, 6, or 12 photos in an images folder nested in a project foler. Here is an example of what it should look like:

```bash
Projects/
├── Scene
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg 
```

InstantSplat comes with example data to use as a test located at:
```bash
assets/
└── sora/
    └── Santorini/
        └── images/
            ├── image1.jpg
            ├── image2.jpg
            └── image3.jpg
    └── Art/
        └── images/
            ├── image1.jpg
            ├── image2.jpg
            └── image3.jpg
```

### Running Inference
The windows implementation currently only supports inference. If you are looking to run eval, refer to the original project page.

#### Using Gradio
Run `python instantsplat_gradio.py`

Once launch, navigate to `http://127.0.0.1:7860/` in your browser.

#### Using CLI
Run `python run_infer.py /path/to/input/images /path/to/output --n_views 3 --iterations 1000`

**Command line arguments:**

**--n_views**
Number of input views. Must be 3, 6, or 9

**--iterations 1000**
Number of training iterations, can be set from 1000 to 30000. Suggested increasing in increments of 1000.

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [DUSt3R](https://github.com/naver/dust3r)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{fan2024instantsplat,
        title={InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds},
        author={Zhiwen Fan and Wenyan Cong and Kairun Wen and Kevin Wang and Jian Zhang and Xinghao Ding and Danfei Xu and Boris Ivanovic and Marco Pavone and Georgios Pavlakos and Zhangyang Wang and Yue Wang},
        year={2024},
        eprint={2403.20309},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
      }
```
