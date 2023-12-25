from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/TrainAI/runs/detect/train/weights/best.pt')
model.predict('/content/410595871_7018774658158592_3574941633874398090_n (1).jpg',save=True,show=True,conf=0.7)