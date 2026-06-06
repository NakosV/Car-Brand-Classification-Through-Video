# Car-Brand-Classification-Through-Video
A car brand classifier that utilizes the speed of **YOLOv8n** and the accuracy of **ResNet-50** to identify brands in different car traffic videos, built as part of a university course assignment.

# Contents of this Project  
This repository contains the code of the project which is the following file:
- **[`Video_Classification.py`](Video_Classification.py)**  

# How the program works
The code itself even though it is one file, it can be split in two parts. The first part being the training of the **ResNet-50 model** on the **CompCars dataset** in order for the model to be able to learn characteristics of specific car brands. The second part is the usage of the **YOLOv8n model** in conjuction with the now **trained ResNet-50 model** on a video with a few distinct and unique techniques. If you want to dive deeper and learn about the inner workings of the code in more detail **[you can click here.](EXPLANATION.md)**

# Useful Links
 - ***Dataset:*** https://www.kaggle.com/datasets/renancostaalencar/compcars
 - ***The video used:*** https://www.pexels.com/video/dynamic-city-highway-with-arched-bridge-32272314/

# Results
You can check the results of the video by clicking right here
