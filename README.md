# 虚拟人说话头生成

# Get Started

## Installation

Tested on Ubuntu 22.04, Pytorch 1.12 and CUDA 11.6，or  Pytorch 1.12 and CUDA 11.3

```python
git clone https://github.com/waityousea/xuniren.git
cd xuniren
```

### Install dependency

```python
# for ubuntu, portaudio is needed for pyaudio to work.
sudo apt install portaudio19-dev

pip install -r requirements.txt
or
## environment.yml中的pytorch使用的1.12和cuda 11.3
conda env create -f environment.yml 
## install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Build extension (optional)

By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime. However, this may be inconvenient sometimes. Therefore, we also provide the `setup.py` to build each extension:

```
# install all extension modules
bash scripts/install_ext.sh
```

### **start**

环境配置完成后，启动虚拟人生成器：

```python
python app.py
```

接口的输入与输出信息 [Websoket.md](https://github.com/waityousea/xuniren/blob/main/WebSocket.md)

虚拟人生成的核心文件

```python
## 注意，核心文件需要单独训练
.
├── data
│   ├── kf.json			
│   ├── pretrained
│   └── └── ngp_kg.pth

```

### Inference Speed

在台式机RTX A4000或笔记本RTX 3080ti的显卡（显存16G）上进行视频推理时，1s可以推理35~43帧，假如1s视频25帧，则1s可推理约1.5s视频。

# Acknowledgement

- The data pre-processing part is adapted from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF).
- The NeRF framework is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).
- The algorithm core come from  [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF).

学术交流可发邮件到邮箱：waityousea@126.com