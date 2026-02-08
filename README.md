# Flower Image Classification using CNN
This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images of flowers into five categories: daisy, dandelion, rose, sunflower, and tulip. The model is trained on the publicly available TensorFlow flower dataset and demonstrates key deep learning concepts such as image preprocessing, data augmentation, model evaluation, and saving.

---

## Dataset Overview
- **Source:** [TensorFlow Flower Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- Total Images: ~3,700
- Categories: daisy, dandelion, rose, sunflower, tulip

---

## Tools & Technologies Used
- Python  
- TensorFlow & Keras  
- NumPy  
- Matplotlib  
- VS Code

---

## Data Preprocessing
- Resized images to **180x180**
- Split dataset into **80% training** and **20% validation**
- Applied **Rescaling (1./255)** for normalization
- Used **data augmentation** (flip, rotation, zoom)
- Batching, shuffling, caching, and prefetching for performance

---

## Model Architecture
```text
Input (180x180x3)
↓
Data Augmentation
↓
Rescaling Layer
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Dropout (0.2)
↓
Flatten
↓
Dense (128) → ReLU
↓
Dense (5 classes)
```
*Optimizer: adam
*Loss: SparseCategoricalCrossentropy(from_logits=True)
*Metrics: accuracy

---

## Results & Visualizations
*Achieved high training and validation accuracy
*Plotted training vs. validation accuracy & loss over 15 epochs
*Added data augmentation to improve model generalization

---

## Model Saving 
model.save('flower_model.keras')

---

## Conclusion
This project demonstrates how CNNs can classify images with high accuracy using TensorFlow. It covers end-to-end deep learning pipeline: preprocessing, model building, training, evaluation, and saving.

---

## Author
Mansi Rajapkar,
BCA Student & Aspiring Data Scientist
GitHub : https://github.com/mansirajapkar | LinkedIn:https://linkedin.com/in/mansi-rajapkar-8ab073343
