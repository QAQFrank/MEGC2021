## FAMGAN

### Requirements

* Python 3
* PyTorch 0.4.1
* visdom (optional, only for training with browser visualizer)
* imageio (optional, only for generating GIF image in testing)

### Train

```
python main.py --data_root [path_to_dataset]

# e.g. python main.py --data_root datasets/celebA --gpu_ids 0,1 --sample_img_freq 500
#      python main.py --data_root datasets/emotionNet --gpu_ids 0,1 --sample_img_freq 500
#      set '--visdom_display_id 0' if you don't want to use visdom
#      use 'python main.py -h' to check out more options.
```

### Test

```
python main.py --mode test --data_root [path_to_dataset] --ckpt_dir [path_to_pretrained_model] --load_epoch [epoch_num]

# e.g. python main.py --mode test --data_root datasets/celebA --batch_size 8 --max_dataset_size 150 --gpu_ids 0,1 --ckpt_dir ckpts/celebA/ganimation/190327_161852/ --load_epoch 30
#      set '--interpolate_len 1' if you don't need linear interpolation.
#      use '--save_test_gif' to generate animated images.
```

## RealSR

This part is based on https://github.com/Tencent/Real-SR

### Usage

- Usage - ```./realsr-ncnn-vulkan -i in.jpg -o out.png```
- ```-x``` - use ensemble 
- ```-g 0``` - select gpu id.