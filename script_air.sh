CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc_air.py --base_lr 0.2 --batch_size 64 --epochs 51 --weight_decay 1e-6 &> air_fc.txt; 
CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_all_air.py --base_lr 2e-2 --batch_size 32 --epochs 50 --weight_decay 1e-3 --model "vgg_16_epoch_51.pth" &> air_all.txt;

