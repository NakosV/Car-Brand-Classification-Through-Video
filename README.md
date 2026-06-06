# Car-Brand-Classification-Through-Video
A car brand classifier that utilizes the speed of **YOLOv8n** and the accuracy of **ResNet-50** to identify brands in different car traffic videos, built as part of a university course assignment.

# Contents of this Project  
This repository contains the code of the project which is the following file:
- **[`Video_Classification.py`](Video_Classification.py)**  

# How the program works
The code itself even though it is one file, it can be split in two parts. The first part being the training of the **ResNet-50 model** on the **CompCars dataset** in order for the model to be able to learn characteristics of specific car brands. The second part is the usage of the **YOLOv8n model** in conjuction with the now **trained ResNet-50 model** on a video. 

## Further Details
###  - Training
As listed before, the **CompCars dataset** was used in this project
