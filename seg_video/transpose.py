import torch
import cv2
import numpy as np

# replace
model = torch.jit.load("unet_ground_plane.pt")
model.eval()

# replace 
cap = cv2.VideoCapture("example_video.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# frame of video config 
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_masked_video.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # Resize and preprocess
    resized = cv2.resize(frame, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(tensor)
    output = output.squeeze().cpu().numpy()

    # Convert to binary mask
    mask = (output > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (width, height))

    color_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0)

 # ----------------------- fit central line in mask -----------------------
    central_points = []
    step = 5  # adjust smoothness vs. speed
    for y in range(height - 1, 0, -step):  # bottom to top
        row = mask_resized[y, :]
        ground_x = np.where(row == 255)[0]
        if ground_x.size > 0:
            avg_x = int(np.mean(ground_x))
            central_points.append((avg_x, y))

    if len(central_points) >= 3:
        x_vals, y_vals = zip(*central_points)
        poly = np.poly1d(np.polyfit(y_vals, x_vals, deg=2))  # quadratic fit

        min_y = min(y_vals)
        max_y = max(y_vals)

        prev_x, prev_y = int(poly(max_y)), max_y
        for y in range(max_y, min_y - 1, -10):
            x = int(poly(y))
            cv2.line(blended, (prev_x, prev_y), (x, y), (153, 51, 102), 2)
            prev_x, prev_y = x, y

    # Draw the central line
    #for i in range(1, len(central_points)):
    #    cv2.line(blended, central_points[i - 1], central_points[i], (255, 153, 204), 2) 
    out.write(blended)

    cv2.imshow("Segmentation Overlay", blended)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

