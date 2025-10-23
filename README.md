# ML-KNN-SVM-K-MEANS

## Overview
This project applies **three machine learning techniques** — KNN, SVM, and K-Means — on a **gene expression dataset (lncRNA_5_Cancers.csv)** that includes five cancer types:
**KIRC, LUAD, LUSC, PRAD, and THCA.**

### Tasks
1. **KNN Classification**
   - 5-fold stratified cross-validation
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix, ROC-AUC, and PR-AUC
2. **SVM Classification**
   - Linear, Polynomial, and RBF kernels
   - Performance comparison across kernels
3. **K-Means Clustering**
   - K = 2–7
   - PCA visualization
   - Elbow & Silhouette methods for optimal K

---

## Run Instructions
```powershell
python .\HW4.py --csv .\data\lncRNA_5_Cancers.csv --out .\results\
```

All output files (figures and summary CSV) will appear inside `results/`.

---

## Requirements
- Python 3.9+
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

Install dependencies:
```powershell
pip install -r requirements.txt
```

---

## Folder Descriptions
| Folder | Description |
|---------|-------------|
| `data/` | Input dataset (not uploaded to repo if large) |
| `results/` | Auto-generated metrics and visualizations |
| `reports/` | Report templates and final documentation |

---

## Author
<Livan Miranda>  
Course: CAP5610 – Machine Learning  
Fall 2025
