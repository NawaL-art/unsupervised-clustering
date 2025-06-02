# 🧬 Unsupervised Clustering on Zoo Dataset

This project applies **agglomerative hierarchical clustering** to classify animals in the Zoo dataset based on their characteristics.

## 🔍 What it does

- Loads and explores the Zoo dataset
- Visualizes the distribution of animal classes
- Uses **Agglomerative Clustering** with cosine similarity
- Compares predicted clusters to actual classes using RMSE

## 📊 Techniques Used

- Agglomerative Clustering (`scikit-learn`)
- Cosine distance metric
- Data preprocessing with `pandas` and `numpy`
- Visualization with `matplotlib`

## 📁 Dataset

- **zoo.data**: Includes 101 animals with 16 features and a class label.

## 🧠 Output Example

- Class distribution bar chart
- RMSE printed for performance comparison (unsupervised vs true labels)

