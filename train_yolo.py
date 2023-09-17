from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='pnid.yaml', batch=4, epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()

wandb.finish()