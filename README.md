Train a CNN that, for each pixel in a color image, predicts “ground” vs. “not-ground.”  No depth sensor needed—just the RGB stream, so it's independent of ROS and there is no need for extra equiptment.

### STEPS: 
1. Data collection & labeling
2. Model selection for CNN & training (segmentation_models_pytorch ?)
3. Inference script for real-time camera detection (?)

### LabelMe for Image labelling (ground vs not ground)

- see GUI and label images: $ labelme
- use toolkit options:
labelmetk [TOOLKIT] [OPTIONS] [FILE_OR_DIR]

• [`ai-annotate-rectangles`](https://labelme.io/docs/ai-annotate-rectangles)
• [`ai-rectangle-to-mask`](https://labelme.io/docs/ai-rectangle-to-mask)
• [`export-to-voc`](https://labelme.io/docs/export-to-voc)
• [`export-to-yolo`](https://labelme.io/docs/export-to-yolo)
• [`extract-image`](https://labelme.io/docs/extract-image)
• [`import-from-yolo`](https://labelme.io/docs/import-from-yolo)
• [`json-to-mask`](https://labelme.io/docs/json-to-mask)
• [`json-to-masks`](https://labelme.io/docs/json-to-masks)
• [`json-to-visualization`](https://labelme.io/docs/json-to-visualization)
• [`list-labels`](https://labelme.io/docs/list-labels)
• [`print-stats`](https://labelme.io/docs/print-stats)
• [`rename-labels`](https://labelme.io/docs/rename-labels)
• [`resize-image`](https://labelme.io/docs/resize-image)

