### 1. Install CUDA Toolkit

Check version CUDA on laptop

```
nvidia-smi
```

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.30                 Driver Version: 546.30       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   36C    P0              13W /  60W |      0MiB /  4096MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

=>  CUDA Version: 12.3 

Install CUDA toolkit version 12.3

### 2. Create virtual environment python

```
python -m venv uav
.\\uav\\Scripts\\active
(venv) ...
```


### 3. Install ltralytics

```
(venv) pip install ultralytics
```

### 4. Install Pytorch 

Link check version with CUDA: https://pytorch.org/get-started/previous-versions/ 

```
CUDA 12.3 → pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 
```

```
>>> import ultralytics
>>> ultralytics.checks()
Ultralytics 8.4.12  Python-3.10.11 torch-2.5.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096MiB)
Setup complete  (16 CPUs, 13.9 GB RAM, 68.6/100.0 GB disk)
```


thấy NVIDIA Geforce RTX là oke, còn nếu thấy CPU thì lỗi
