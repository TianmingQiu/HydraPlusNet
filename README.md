# Hydraplus_Net
A re-implementation of [HydraPlus Net](https://arxiv.org/abs/1709.09930) based on Pytorch.  
With the help of (caffe implementtion)(https://github.com/xh-liu/HydraPlus-Net) from original authors and another [pytorch version](https://github.com/CCC-123/Hydraplus_Net).
The network structure is the same as the [caffe version](https://github.com/xh-liu/HydraPlus-Net/tree/master/prototxt_example). 

## Requirements  
pytorch  
CUDA 8.0  

## Dataset  
- PA-100K dataset  
- Download [link](https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M)
- The directory structure looks like:
```shell script
|--Hydraplus  
|----data  
|------PA-100K  
|--------annotation  
|--------ralease_data  
|----------release_data  
```


## Train
We train the network stagewisely, you can start with MNet:
```shell script
CUDA_VISIBLE_DEVICES=6,7 python train.py -m MNet -bs 128 -lr 0.01 -nw 16 -mGPUs
```
And then we train the attention branch with MNet weights:
```shell script
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m AF2 -mpath checkpoint/MNet_epoch_995 -bs 128 -lr 0.01 -nw 16 -mGPUs
```

## Test
```shell script
python test.py -m AF1 -p checkpoint/AF1_epoch_0
```
