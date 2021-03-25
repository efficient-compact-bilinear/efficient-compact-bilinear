CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc_dtd.py --base_lr 1 --batch_size 64 --epochs 51 --weight_decay 1e-6 &> dtd_fc.txt; 
CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_all_dtd.py --base_lr 1e-2 --batch_size 64 --epochs 50 --weight_decay 1e-3 --model "vgg_16_epoch_51.pth" &> dtd_all.txt;

