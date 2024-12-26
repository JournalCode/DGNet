# DGNet: Enhancing Parallel CTR Prediction Models via Decoupled Gated Network
## The structure of DGNet. 
![image](./images/DENet.jpg)
The structure of DGNet with DCNv2 as the backbone model. DGNet contains two core components: the decoupled embedding generator (DEG) and the gate fusion network (GateF).  

## Requirements
python==3.10.8

torch==2.3.0+cu121

### Run the code


```CUDA_VISIBLE_DEVICES=0 python run_base.py```


## Experiment results
![image](./images/DEG.png)


![image](./images/GateF.png)

