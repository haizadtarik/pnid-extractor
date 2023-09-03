from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='pnid.yaml', batch=4, epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()
