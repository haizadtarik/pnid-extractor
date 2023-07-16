import requests
from PIL import Image
import torch
from transformers import AutoProcessor, OwlViTForObjectDetection
import numpy as np
import cv2

processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
image = Image.open('pnid.jpg')
query_image = Image.open('tx.jpg').convert('RGB') 
inputs = processor(images=image, query_images=query_image, return_tensors="pt")
with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)
# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_image_guided_detection(
    outputs=outputs, threshold=0.6, nms_threshold=0.5, target_sizes=target_sizes
)
img = None
i = 0  # Retrieve predictions for the first image
boxes, scores = results[i]["boxes"], results[i]["scores"]
for box, score in zip(boxes, scores):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
    img = np.asarray(image)
    for box, score in zip(boxes, scores):
        box = [int(i) for i in box.tolist()]

        if score >= 0.6:
            img = cv2.rectangle(img, box[:2], box[2:], (0,255,0), 5)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25
if img is not None:
    output_image = Image.fromarray(img)
    output_image.show()