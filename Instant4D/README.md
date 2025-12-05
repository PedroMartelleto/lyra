# Instant4D: 4D Gaussian Splatting in Minutes

<a href="https://instant4d.github.io/"><img src='https://img.shields.io/badge/Website-Instant4D-green' alt='Website'></a>
<a href="https://arxiv.org/abs/2510.01119"><img src='https://img.shields.io/badge/PDF-Instant4D-yellow' alt='PDF'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-Instant4D-blue' alt='Paper BibTex'></a>



## Installation

Clone the repository with the submodules by using:
```shell
git clone --recursive git@github.com:Zhanpeng1202/Instant4D.git
```

### Environment

Update requirements.txt with correct CUDA version for PyTorch and cuUML, i.e., replacing cu126 and cu12 with your CUDA version.
```shell
conda create -n instant4d python=3.10
conda activate instant4d
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # change to your CUDA version 
pip install -r requirement.txt


pip install xformers # read below
# Note: mega-sam requires xformers for Unidepth, but there is a dependancy issue for newer pytroch
# But Gaussian Splatting require CUDA version match.
# therefore one workaround is to use a different virtual environment only for Unidepth metric depth estimaiton
```


To install mega-sam, run the following command: <br>
Note: change the `.type()` to `scalar_type()` in `mega-sam\base\src\altcorr_kernel`, `mega-sam\base\src\correlation_kernels` and `mega-sam/base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu` if using torch >2.7, refer this [issue](https://github.com/NVIDIAGameWorks/kaolin/issues/865).
```shell
cd SLAM/mega-sam/base
python setup.py install
cd ../../../../
```

To install Gaussian Splatting accelerating package, run the following command: 

```shell
cd submodule
pip install fussed-ssim
pip install simple-knn
cd pointops2
python setup.py install
cd ../..
```
Noted that the gaussian splatting package will be compile during the first running.


### Downloading pretrained checkpoints for mega-sam

1.  Download [DepthAnything checkpoint](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) to
    mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth

2.  Download and include [RAFT checkpoint](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) at mega-sam/cvd_opt/raft-things.pth



### 4DGS Remote Viewer

We provide a lightweight websocket remote viewer to visualize 4DGS training process. Users can train 4DGS on a server and hope to view it on local computer.

On the local computer

```shell
# download these file in local computer
git clone git@github.com:Zhanpeng1202/gaussian_splatting_websocket_viewer.git

# Connect Server with SSH with vscode
vscode ssh server 

#set up forward port in vscode
Terminal -> Ports -> Forward a Ports -> 6119
```

On the server

```shell
# clone the official gaussain splatting repository
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive

# put networkGUI_Websocket.py in to correct location inside the cloned repository
<location>
|---gaussian_splatting
|   |---gaussain_render
|   |   |---network_gui.py
|   |   |---network_gui_websocket.py

|   |---train.py 
# replace train.py with that provided in this repository
```





### Dataset

```
mkdir dataset
cd dataset
```

### [Nvidia](https://gorokee.github.io/jsyoon/dynamic_synth/)

Download the pre-processed data by [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF).
```
mkdir Nvidia
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
unzip data.zip
rm data.zip
```

### [DAVIS](https://davischallenge.org/davis2016/code.html) or custom sequences
We provide sample videos under `examples/`, one can start from reproduce them.



### Optimization

Change the `path` and `weight` in `script/reconstruct.sh` and change the config accordingly in the script/optmize
```
source script/reconstruct.sh 
python -m script.prune
python -m script.optmize
```


## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Mega-SAM](https://github.com/mega-sam/mega-sam)

## Citation

If you find this project useful in your research, please consider citing:
```
@article{luo2025instant4d,
  title={Instant4d: 4d gaussian splatting in minutes},
  author={Luo, Zhanpeng and Ran, Haoxi and Lu, Li},
  journal={Advances in neural information processing systems},
  year={2025}
}
```
