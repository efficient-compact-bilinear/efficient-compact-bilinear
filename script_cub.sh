CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc_cub.py --base_lr 1 --batch_size 64 --epochs 81 --weight_decay 1e-6 &> cub_fc.txt; 
CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_all_cub.py --base_lr 1e-2 --batch_size 32 --epochs 50 --weight_decay 1e-3 --model "vgg_16_epoch_81.pth" &> cub_all.txt;

