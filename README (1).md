# Anomaly Detection on CIFAR-10 using Deep One-Class Classification

This project implements an anomaly detection system using a simplified Deep SVDD approach on the CIFAR-10 dataset. One class ("airplane") is treated as the normal class, while all others are considered anomalies.

The project was developed as a final assignment for the BLG454E - Learning From Data course (Spring 2025) at Istanbul Technical University.

---

##  Project Structure

- `deep_svdd_cifar10.ipynb` → Jupyter notebook with all implementation steps
- `BLGFinal_project.pdf` → Final report in IEEE format
- `roc_comparison.png` → ROC curve comparing Deep SVDD vs One-Class SVM
- `pca_visualization.png` → 2D projection showing feature space clustering

---

##  What This Project Does

- Filters only "airplane" images from CIFAR-10 for training
- Extracts features via PCA
- Trains a Deep SVDD-style anomaly detector using center-distance
- Trains a baseline One-Class SVM model
- Compares both using accuracy, precision, recall, and ROC-AUC
- Visualizes results with matplotlib (PCA and ROC curve)

---

##  Results Summary

| Model               | Accuracy | Precision | Recall | AUC  |
|--------------------|----------|-----------|--------|------|
| Deep SVDD (center) | 89.3%    | 88.2%     | 87.9%  | ~0.91 |
| One-Class SVM      | 75.6%    | -         | -      | ~0.78 |

---

##  Presentation Video

A short video presentation (3 minutes) explaining the project and results is available [here](#).

>  Replace the link above with your YouTube or Drive link

---

##  Requirements

This notebook requires the following packages:
- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `numpy`

You can install them using:
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

---

##  Author

**Rabia Güneş**  
150210341 – Artificial Intelligence and Data Engineering  
Istanbul Technical University
