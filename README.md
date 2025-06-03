## Robotic Vision Project: 
Train convolutional neural networks for ground-plane segmentation and navigation planning. Model specified and fine-tuned for hospital environment use (hospital specific navigation).

## Steps: 
1. Data collection & labeling using labelme
2. Model selection for CNN & training
3. Inference script and fine-tuning 
4. Adapt script for video use: real robotic vision prototype 

## Order of scripts: 
1. convert_masks.py (make sure all file directories are correct)
2. split.sh
3. train.py (-mode train for training or -mode infer for specific checkpoint)

Visualization of masks produced by models: 
4. x_metrics.sh infile outfile 
5. plot.py

Video Inference script and visualization: 
6. extract_best_model.py
7. transpose.py (make sure all file directories are correct)  

### current metrics can be found under visualisations

