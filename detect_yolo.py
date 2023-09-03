from ultralytics import YOLO
from PIL import Image
model = YOLO('./weights/best.pt')
results = model('images/image6.png')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image