# **Robotic Vision Project:**

This project implements a model that performs **ground segmentation** using a U-Net model and determines a robot's **directional movement decision** based on obstacle-free zones. The U-Net model with a MobileNetV2 encoder is trained to distinguish traversable ground from obstacles in an environment. Once trained, the model can be deployed across static images, prerecorded videos, or live webcam streams. The segmented output is processed through a "laser beam" logic that divides the image into horizontal regions, scores each based on pixel by pixel ground coverage, and determines the optimal motion direction — such as turning left, right, or proceeding straight—based on which region is safest. This decision is communicated as degrees, and a line in the corresponding direction is drawn. Furthermore, the motion_decision script also outputs and saves a video showcasing the classification probability as a 'heatmap', where the red zones are the areas the model classifies most confidently as safe. This approach is purely vision-based, portable, and computationally lightweight - it might be useful for prototyping computer vision for robotic navigation on simple laptops. 

**Dataset**: Custom-labeled images with binary ground masks - using ‘Labelme’

**Model**: U-Net (semantic segmentation: segmentation_models_pytorch)

**Backbone**: MobileNetV2 (lightweight, efficient)

**Optimizer**: Adam

**Main libraries used**: PyTorch, TorchScript, OpenCV, Matplotlib, Albumentations


## **Order of scripts:**

1. convert_masks.py (make sure all file directories are correct)
2. split.sh
3. train.py (-mode train for training or -mode infer for specific checkpoint)

### **Visualization of masks produced by models:**

1. x_metrics.sh infile outfile
2. plot.py

### **Video Inference script and visualization:**

1. extract_best_model.py
2. transpose.py (make sure all file directories are correct)

### **Live Feed Inference script and visualization:**

1. extract_best_model.py
2. either motion_decision.py for simpler ML and CV integration (future improvements) or motion_decision.cpp for faster inference and decision

#### Motion Direction with predicted ground-plane mask (screenshot)
![alt text](image-1.png)

#### Motion Direction with classification prediction (screenshot)
![alt text](image-2.png)

#### **some more current metrics and output images/ videos can be found under visualisations**
