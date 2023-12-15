# Wise_tooth_detection
# Project Title: Wise Tooth Classification in Panoramic Radiographs
Description
This project aims to develop a convolutional neural network (CNN) for classifying panoramic radiographs into two categories: presence or absence of wise teeth.

# Dataset and Data Preprocessing
Used a dataset of panoramic radiographs with labeled wise teeth from google drive - https://drive.google.com/drive/folders/1HrDvSiFMGKefq_75KR9VKERq1CTeZFZ6?usp=drive_link
Preprocessed images by resizing to (224, 224), normalizing intensity, and applying data augmentation (random shearing, zooming, and flipping).

# Model Architecture
Sequential CNN with three convolutional blocks, each followed by max pooling and dropout.
Each block consists of a 64-filter convolution layer with ReLU activation.
Input shape: (224, 224, 1), grayscale.
Output layer: 2 neurons with softmax activation for binary classification.

# Training and Evaluation
Trained the model for 10 epochs with Adam optimizer and learning rate 0.0005.

# Code Structure and Dependencies
Code organized into separate scripts for data preprocessing, model building, and evaluation.
Requires TensorFlow, NumPy, scikit-learn, and other libraries.

# Contributions and Future Work
Demonstrated the feasibility of using CNNs for classifying wise teeth in panoramas.

# Potential future work includes:
Expanding the dataset with more diverse cases.
Exploring different CNN architectures and optimization techniques.
