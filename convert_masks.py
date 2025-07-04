import os
from PIL import Image
from dataset import json_to_mask

IMG_DIR  = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment"
JSON_DIR = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment/hospital_json"
MASK_DIR = "/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/train_masks"
os.makedirs(MASK_DIR, exist_ok=True)

for fn in os.listdir(JSON_DIR):
    if not fn.endswith(".json"): continue
    base = fn[:-5]  # strip “.json”
    img = Image.open(f"{IMG_DIR}/{base}.png")     
    w,h = img.size                                  # note: (width, height)
    
    ground_mask, not_ground_mask = json_to_mask(f"{JSON_DIR}/{fn}", (w, h)) # extract json ground_mask and not_ground_mask
    final_mask = ground_mask & (~not_ground_mask.astype(bool)) # remove obstacles within ground_mask

    Image.fromarray(final_mask.astype("uint8") * 255).save(f"{MASK_DIR}/{base}.png")


# Each call produces exactly one mask file (base.png), which lines up with one image file (base.jpg).
