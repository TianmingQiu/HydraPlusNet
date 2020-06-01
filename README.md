# Hydraplus_Net

an re-implementation of HydraPlus Net based on Pytorch.  
MNet uses Inception_v3.  

## requirement  
pytorch  
visdom  
CUDA 8.0  
## dataset  
PA-100K dataset  
Download [link](https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M)

|--Hydraplus  
|----data  
|------PA-100K  
|--------annotation  
|--------ralease_data  
|----------release_data  

## 1.train  
```shell script
CUDA_VISIBLE_DEVICES=6,7 python train.py -m MNet -bs 4 -lr 0.01 -nw 4 -mGPUs
```

