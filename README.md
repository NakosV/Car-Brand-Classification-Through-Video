# Car-Brand-Classification-Through-Video
A car brand classifier that utilizes the speed of **YOLOv8n** and the accuracy of **ResNet-50** to identify brands in different car traffic videos, built as part of a university course assignment.

# Contents of this Project  
This repository contains the code of the project which is the following file:
- **[`Video_Classification.py`](Video_Classification.py)**  

# How the program works
The code itself even though it is one file, it can be split in two parts. The first part being the training of the **ResNet-50 model** on the **CompCars dataset** in order for the model to be able to learn characteristics of specific car brands. The second part is the usage of the **YOLOv8n model** in conjuction with the now **trained ResNet-50 model** on a video with a few distinct and unique techniques. If you want to dive deeper and learn about the inner workings of the code in more detail **[you can click here.**](EXPLANATION.md) 

## Further Details
###  1. Preprocessing
As listed before, the **CompCars dataset** was used in this project but due to its huge size and uneven samples of car images per brand some clean up was required before the training. I mainly had to reorganize the files in folders with the actual names of the brands and after that, I deleted any folders that had under 800 pictures of that brand because I believe that, anything with that amount of pictures and under, where to small for the model to properly learn.
### 2. Training
As mentioned before the pretrained model **ResNet-50** was used for the classification part of this program. I trained it especially on the cleaned up dataset. The training itself can be broken up to two parts. The first is the training of only the fully connected the layer. This is done for 10 epochs with a learning rate of 0.001. This step has the purpose of not disturbing the main abilities and knowledge of the model, and only training it on the brands of my dataset. The second part of the training is more intense. In this part, the last two layers of the model are unfrozen and trained on my data with the hopes that the model will learn a lot of the small details of the cars, like particular shapes or small characteristics. This step lasts for 30 epochs and has a learning rate of 0.0001
### 3. Inference
In this part of the program, both of the models are utilized. I load **my trained ResNet-50 model** and use it with the **YOLOv8n**. Because of the video that I used, I utilized a unique technique. I draw two lines on the screen. One on the 60% mark of the screen named ***classify_line_position*** and another on the 85% mark of the screen named ***count_line_position***. The first is where YOLO starts taking pictures of the car every second frame and the ResNet tries to identify the car brand based on those pictures. The second line is when the cars are actually counted and registered in the system. This way the program has a lot of time to correctly identify a car as it is approching the line and gets closer to the camera. Lastly in order for the program to keep track of a car in between all the frames I used ***Centroid Tracking*** where basically if the centroid of a car moves over a given distance between two frames then it is a different vehicle
