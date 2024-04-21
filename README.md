# Industrial Safety Helmet Recognition

## Introduction

This project aims to detect individuals wearing or not wearing industrial safety helmets in images. The model can mark regions and classify individuals based on the presence of safety helmets.

## Input

Images containing individuals with or without safety helmets.

## Output

For each input image, the model provides region marking and classification results indicating whether individuals are wearing safety helmets or not.

## Installation

Ensure you have Python installed along with the required packages:

Usage
Clone the repository:
bash
Copy code
git clone https://github.com/luongphuongtech/AI_detection.git
cd AI_detection
Run the model with a specific image:
python
Copy code
from yolov8 import YOLO

# Load the pre-trained model
model = YOLO('best.pt')

# Specify the path to the image
image_path = 'path_to_image.jpg'

# Detect and display the results
model.detect(image_path, show=True)
Replace 'best.pt' with the path to your trained model checkpoint.

Training Results
Training results are summarized below:
![image](https://github.com/luongphuongtech/AI_detection/assets/121532605/c9681bc2-3ca8-4b96-921f-fb042a0aac6c)

Training Results

| Class   | Number of Objects | Precision | Recall | mAP50 | mAP50-95 |
|---------|-------------------|-----------|--------|-------|----------|
| ALL     | 5319              | 0.929     | 0.892  | 0.940 | 0.640    |
| Hat     | 864               | 0.936     | 0.910  | 0.944 | 0.742    |
| Person  | 4455              | 0.921     | 0.874  | 0.935 | 0.537    |
