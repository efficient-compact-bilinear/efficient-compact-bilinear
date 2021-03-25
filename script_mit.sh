CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc_mit.py --base_lr 1 --batch_size 64 --epochs 71 --weight_decay 1e-6 &> mit_fc.txt; 
CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_all_mit.py --base_lr 1e-2 --batch_size 32 --epochs 60 --weight_decay 1e-3 --model "vgg_16_epoch_71.pth" &> mit_all.txt;

