## README
For a better user experience, we recommend you to directly use github repository
github [link](https://github.com/efficient-compact-bilinear/efficient-compact-bilinear)



## Prepare datasets
### Dataset
AIR [link](https://drive.google.com/file/d/1KEPS8ZDQD7cNQW8cg1tNN0_Pvq0wSRBG/view?usp=sharing)
CUB [link](https://www.dropbox.com/s/dsgngcy3fmamgm7/cub200.tar.gz?dl=0)
MIT [link](https://www.dropbox.com/s/n6ymftqk8alihpu/mit.tar.gz?dl=0)
DTD [link](https://www.dropbox.com/s/co5lq11axokwkcj/dtd.tar.gz?dl=0)




please unzip the downloaded datasets and then move them to data


## Training

Step 1. Fine-tune the fc layer only.

CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc_air.py --base_lr 0.2 --batch_size 64 --epochs 51 --weight_decay 1e-6
    
Step 2. Fine-tune all layers.

CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_all_air.py --base_lr 2e-2 --batch_size 32 --epochs 50 --weight_decay 1e-3 --model "vgg_16_epoch_51.pth"



## Run Script


You can also run the experiments on four datasets by one script in background:
   
    AIR:
    nohup bash script_air.sh &

    CUB:
    nohup bash script_cub.sh &

    MIT:
    nohup bash script_mit.sh &

    DTD:
    nohup bash script_dtd.sh &

## Acknowledgement

This project is developped based on [bilinear-cnn](https://github.com/HaoMood/bilinear-cnn)
