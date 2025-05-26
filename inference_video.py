import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='/content/drive/MyDrive/Train_SSD300_VGG16_Model_from_Torchvision_on_Custom_Dataset/data/inference_data/video_1.mp4'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = vars(parser.parse_args())

os.makedirs('inference_outputs/videos', exist_ok=True)

COLORS = [
    [0, 0, 0],       # Background (thường không sử dụng)
    [255, 0, 0],     # Lớp 1: Màu đỏ
    [0, 255, 0],     # Lớp 2: Màu xanh lá
    [0, 0, 255],     # Lớp 3: Màu xanh dương
    [255, 255, 0],   # Lớp 4: Màu vàng
    [255, 0, 255]    # Lớp 5: Màu hồng
]

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Define the detection threshold.
detection_threshold = args['threshold']

cap = cv2.VideoCapture(args['input'])

if not cap.isOpened():
    print('Error while trying to read video. Please check path again')
    exit()

# Get the frame width and height.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
print(save_name)
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

# Read until end of video.
while cap.isOpened():
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if args['imgsz'] is not None:
            image = cv2.resize(image, (args['imgsz'], args['imgsz']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Make the pixel range between 0 and 1.
        image /= 255.0
        # Bring color channels to front (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Convert to tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        # Get the start time.
        start_time = time.time()
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        end_time = time.time()
        
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1
        
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            draw_scores = scores[scores >= detection_threshold]  # Lưu lại độ tin cậy tương ứng
            # Get all the predicted class names.
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # Draw the bounding boxes and write the class name and confidence on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                confidence = draw_scores[j]  # Lấy độ tin cậy của dự đoán

                color = COLORS[CLASSES.index(class_name)]
                # Rescale boxes.
                xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                cv2.rectangle(frame,
                              (xmin, ymin),
                              (xmax, ymax),
                              color[::-1], 
                              3)
                # Thêm tên lớp và độ tin cậy lên hình
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, 
                            label, 
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            color[::-1], 
                            2, 
                            lineType=cv2.LINE_AA)

        # Save the result to video file
        out.write(frame)

    else:
        break

# Release VideoCapture().
cap.release()
# Release VideoWriter.
out.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

print('Save outputs in: inference_outputs/videos')
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")