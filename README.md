<h1 align="center">Car Brand Classification Through Video</h1>

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=Ultralytics&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

A hybrid computer vision pipeline that utilizes the speed of **YOLOv8n** for real-time object detection and the accuracy of **ResNet-50** for precise brand classification across dynamic traffic videos. Developed as part of a university machine learning course assignment.

---

## Visual Demonstration

<div align="center">
  <img width="700" alt="Detection Example" src="https://github.com/user-attachments/assets/13c54ce7-6fb1-4466-aa59-ab1402d94ba6" />
</div>

> [!TIP]
> To view the full high-resolution output of the program in action, **[watch the complete results video here](https://drive.google.com/file/d/10hYeSIOSfufaInFkyWj55LAPNFDOUFWv/view?usp=drive_link)**.

---

## How The Architecture Works

The system operates through a two-stage pipeline, encapsulated within [`Video_Classification.py`](Video_Classification.py):

1. **Feature Extraction & Training:** The **ResNet-50** architecture is fine-tuned on the CompCars dataset, allowing the model to learn and isolate the unique topological features of various car brands.
2. **Detection & Tracking Pipeline:** **YOLOv8n** scans the video frames to detect vehicles. Once detected, the system utilizes a custom centroid-tracking algorithm to monitor the cars across frames. As a vehicle crosses a predefined classification line, the cropped region of interest is passed to the trained ResNet-50 model for brand identification.

> [!NOTE]
> If you wish to dive deeper into the code architecture, the thresholding logic, and the math behind the centroid tracking, read the **[Comprehensive Explanation Guide](EXPLANATION.md)**.

---

## Installation & Setup

To run this project locally, ensure you have Python installed. The pipeline requires specific versions of PyTorch to leverage GPU acceleration efficiently.

**1. Install PyTorch with CUDA support**  
*(If you are running on a GPU, use the index provided below. Otherwise, install standard PyTorch).*
```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

Ensure your video and dataset paths are correctly configured inside the script. Run the main pipeline using:
```bash
python Video_Classification.py
```
The script will automatically detect if a trained model exists. If it does not, it will initiate the training sequence before proceeding to video inference.

---

## Useful Resources & Data

- **Model Training Data:** [CompCars Dataset (Kaggle)](https://www.kaggle.com/datasets/renancostaalencar/compcars)
- **Inference Video:** [Dynamic City Highway (Pexels)](https://www.pexels.com/video/dynamic-city-highway-with-arched-bridge-32272314/)

---

## Contributions

While the core academic requirements for this project are complete, contributions are welcome. Feel free to suggest performance improvements, address active issues, or fork the repository to experiment with your own tracking logic or custom datasets.

---

> [!IMPORTANT]
> **Explore More of My Work**  
> This project is part of a broader collection of my academic machine learning implementations. To explore LLMs built from scratch, Generative AI (GANs), and Reinforcement Learning agents, check out my **[University Machine Learning Projects Catalog](https://github.com/NakosV/University-Machine-Learning-Projects-Catalog)**.
