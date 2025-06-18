import time
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from collections import defaultdict
import matplotlib.pyplot as plt

class GroundSegmenter:
    def __init__(self, ckpt_path, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.jit.load(ckpt_path, map_location=self.device)
        self.model.eval()

    def predict_mask(self, img_bgr):
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (320, 320))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.model(tensor).squeeze().cpu().numpy()
        mask = (output > 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h))

class MotionDeciderWithLasers:
    def __init__(self, segmenter, img_width, img_height, num_beams=5):
        self.segmenter = segmenter
        self.width = img_width
        self.height = img_height
        self.num_beams = num_beams
        angle_step = 180 // (self.num_beams - 1)
        self.regions = [str(-90 + i * angle_step) for i in range(self.num_beams)]

    def laser_scan(self, mask):
        laser_scores = defaultdict(int)
        step = self.width // self.num_beams
        for i, region in enumerate(self.regions):
            x_start = i * step
            x_end = x_start + step
            region_mask = mask[self.height//2:, x_start:x_end]  # bottom half only
            laser_scores[region] = np.sum(region_mask == 255)
        return laser_scores

    def decide(self, laser_scores, threshold_ratio=0.05):
        total = sum(laser_scores.values())
        if total == 0:
            return "stop"
        best_region = max(laser_scores.items(), key=lambda x: x[1])[0]
        if laser_scores[best_region] < total * threshold_ratio:
            return "stop"
        return best_region

def process_image(image_path, segmenter, decider):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not load image.")

    mask = segmenter.predict_mask(frame)
    laser_scores = decider.laser_scan(mask)
    decision = decider.decide(laser_scores)

    # visualization base
    blended = cv2.addWeighted(frame, 0.7, cv2.applyColorMap(mask, cv2.COLORMAP_JET), 0.3, 0)
    blended_copy = blended.copy()

    # grayscale-shaded laser beam regions based on normalized score
    step = decider.width // decider.num_beams
    max_score = max(laser_scores.values()) if laser_scores else 1
    h = decider.height
    region_centers = {}
    for i, region in enumerate(decider.regions):
        x_start = i * step
        x_end = x_start + step
        norm_score = laser_scores[region] / max_score if max_score > 0 else 0
        intensity = int(255 * (1 - norm_score))  # dark = high score, light = low
        overlay = blended_copy.copy()
        cv2.rectangle(overlay, (x_start, h//2), (x_end, h), (intensity, intensity, intensity), -1)
        blended_copy = cv2.addWeighted(overlay, 0.3, blended_copy, 0.7, 0)
        cv2.rectangle(blended_copy, (x_start, h//2), (x_end, h), (100, 100, 100), 1)
        score_text = f"{laser_scores[region]}"
        cv2.putText(blended_copy, score_text, (x_start + 5, h - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (230, 230, 230), 1)
        region_centers[region] = (x_start + x_end) // 2


    # direction line
    if decision != "stop" and decision in region_centers:
        center_x = region_centers[decision]
        start_point = (decider.width // 2, h)
        end_point = (center_x, h//2)
        cv2.line(blended_copy, start_point, end_point, (255, 255, 255), 2)

    plt.imshow(cv2.cvtColor(blended_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Motion Decision: {decision}°")
    plt.axis('off')
    plt.show()

    print("Laser scores:", dict(laser_scores))
    print("Final decision:", decision + "°")


if __name__ == "__main__":
    ckpt_path = "unet_ground_plane.pt"

    last_decision_time = 0
    decision_interval = 3  # seconds

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))  # save video
    segmenter = GroundSegmenter(ckpt_path, device='cpu')
    decider = MotionDeciderWithLasers(segmenter, img_width=w, img_height=h, num_beams=7)
    mask = np.zeros((h, w), dtype=np.uint8)  # initialize dummy mask
    laser_scores = {region: 0 for region in decider.regions}  # initialize empty scores
    decision = "waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        decider.width = w
        decider.height = h

        current_time = time.time()
        if current_time - last_decision_time >= decision_interval:
            mask = segmenter.predict_mask(frame)
            laser_scores = decider.laser_scan(mask)
            decision = decider.decide(laser_scores)
            last_decision_time = current_time

        blended = cv2.addWeighted(frame, 0.7, cv2.applyColorMap(mask, cv2.COLORMAP_JET), 0.3, 0)
        blended_copy = blended.copy()

        step = decider.width // decider.num_beams
        max_score = max(laser_scores.values()) if laser_scores else 1
        region_centers = {}

        for i, region in enumerate(decider.regions):
            x_start = i * step
            x_end = x_start + step
            norm_score = laser_scores[region] / max_score if max_score > 0 else 0
            intensity = int(255 * (1 - norm_score))
            overlay = blended_copy.copy()
            cv2.rectangle(overlay, (x_start, h//2), (x_end, h), (intensity, intensity, intensity), -1)
            blended_copy = cv2.addWeighted(overlay, 0.3, blended_copy, 0.7, 0)
            cv2.rectangle(blended_copy, (x_start, h//2), (x_end, h), (100, 100, 100), 1)
            score_text = f"{laser_scores[region]}"
            cv2.putText(blended_copy, score_text, (x_start + 5, h - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (230, 230, 230), 1)
            region_centers[region] = (x_start + x_end) // 2

        if decision != "stop" and decision != "waiting..." and decision in region_centers:
            center_x = region_centers[decision]
            start_point = (decider.width // 2, h)
            end_point = (center_x, h//2)
            cv2.line(blended_copy, start_point, end_point, (255, 255, 255), 2)

        cv2.imshow("Motion Decision", blended_copy)
        out.write(blended_copy)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
