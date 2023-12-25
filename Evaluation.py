from ultralytics import YOLO
#Evaluation model
model = YOLO('yolov8x.pt')  # load an official model
model = YOLO('path/to/your/model')
metrics=model.val(data='path/to/your/file/data.yaml')


 #Draw result of processing train.
import pandas as pd
import matplotlib.pyplot as plt

# Function to create a plot for each column
def plot_column(data, column_name):
    plt.figure(figsize=(3, 4))
    plt.plot(data['epoch'], data[column_name], marker='o', linestyle='-', color='b')
    plt.title(column_name)
    plt.xlabel('Epoch')
    plt.ylabel(column_name)
    plt.grid(True)
    plt.show()

# Load the CSV file
file_path = '/content/results (3).csv'
data = pd.read_csv(file_path)

# Strip leading spaces from column names
data.columns = data.columns.str.strip()

# Plotting each column in the dataframe, except for 'epoch' which is the x-axis
for col in data.columns:
    if col != 'epoch':
        plot_column(data, col)