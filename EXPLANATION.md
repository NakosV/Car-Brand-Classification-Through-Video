<h1 align="center">Comprehensive Architecture & Logic Guide</h1>

This document provides an in-depth breakdown of the pipeline, detailing how the data was managed, how the deep learning models were trained, and the spatial tracking logic used during video inference.

---

## 1. Dataset Curation & Preprocessing

The **CompCars dataset** serves as the foundation for this project. Due to its massive scale and severe class imbalances, strict data curation protocols were applied prior to training:

*   **Directory Restructuring:** Raw data was reorganized into standard hierarchical folders, mapped directly to specific car brand names to ensure compatibility with PyTorch data loaders.
*   **Class Filtering:** To prevent overfitting and ensure the model had enough variance to learn robust features, any brand containing fewer than **800 images** was purged from the dataset.

## 2. Two-Phase Transfer Learning

Classification relies on a pre-trained **ResNet-50** architecture. To adapt this model to the specific topological features of car brands without triggering catastrophic forgetting, the training was split into two distinct phases:

> **Phase 1: Classifier Adaptation**
> *   **Duration:** 10 Epochs | **Learning Rate:** 0.001
> *   **Mechanics:** The entire convolutional backbone is frozen. Only the newly initialized Fully Connected (FC) layer is trained. This allows the model to map its existing, generalized visual knowledge to the new car brand classes safely.

> **Phase 2: Deep Fine-Tuning**
> *   **Duration:** 30 Epochs | **Learning Rate:** 0.0001
> *   **Mechanics:** The last two convolutional blocks of ResNet-50 are unfrozen. The network is trained with a heavily reduced learning rate to learn fine-grained, domain-specific details (e.g., grill shapes, headlight contours, logo placements).

## 3. Inference & Spatial Tracking Logic

The inference stage merges **YOLOv8n** (object detection) with the custom-trained **ResNet-50** (classification). To optimize accuracy on moving video footage, the system employs a dual-line spatial trigger mechanism:

*   **The Identification Zone (`classify_line_position` at 60%):** Once a vehicle crosses this upper threshold, YOLOv8n triggers a crop of the bounding box every two frames. These crops are passed to ResNet-50 for continuous prediction as the car gets larger and clearer approaching the camera.
*   **The Registration Zone (`count_line_position` at 85%):** The vehicle is officially logged and counted only when it hits this lower threshold. This buffer zone ensures the classifier has ample time and multiple frames to finalize a high-confidence prediction.
*   **Centroid Tracking:** To maintain vehicle identity between frames, the system calculates the distance between bounding box centroids. If a centroid remains within a defined pixel radius across consecutive frames, it is successfully tracked as the exact same vehicle.

## 4. Fault Tolerance (Checkpointing)

Given the computational intensity of deep fine-tuning, the script incorporates automated checkpointing. Model states, optimizer weights, and learning rate schedulers are periodically saved. This guarantees that training progress is strictly preserved and can be seamlessly resumed in the event of hardware failure or a system crash.

---

> [!NOTE]
> **Performance & Hardware Dependencies**
> On local hardware, the complete execution (predominantly Phase 2 of training) took approximately 5 hours. System performance, inference FPS, and overall accuracy scale directly with hardware capabilities and dataset quality. Users are highly encouraged to experiment with the hyperparameters, spatial thresholds, and custom datasets to optimize the pipeline for their specific use cases.
