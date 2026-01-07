# Satellite Imagery-Based Property Valuation

**Author:** Piyush Sagatani  
**Project:** Multimodal regression pipeline to predict property prices using tabular data + satellite imagery.  
**Submission:** 7 January 2026

---

## Project Flow (end-to-end)
1. **Data Fetcher** (`data_fetcher.ipynb`) — script to download satellite images from Mapbox (commented placeholder present since images already downloaded).  
2. **Preprocessing** (`preprocessing.ipynb`) — data cleaning, missing-value handling, feature engineering (land_usage_ratio, privacy_score, zip_wealth_rank), scaling and saving artifacts.  
3. **Feature Extraction** (`feature_extraction.ipynb`) — Image features and CNN embeddings (ResNet18) saved to `artifacts/`.  
4. **Model Training** (`model_training.ipynb`) — train tabular-only models and tabular+image (multimodal) models, metric comparison, and saving best pipeline to `artifacts/`.  
5. **Submission** (`submission.ipynb`) — generate final predictions CSV (submission file).  
6. **Inference & Explainability** (`inference.ipynb`) — single-point inference example.

---

## Repo layout
```
project/
├── data/processed             # The final clean training data 
├── images_mapbox/             # satellite images (already downloaded)
├── artifacts/                 # embeddings, scalers, models, CSVs (generated)
├── notebooks/
    ├── data_fetcher.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   ├── submission.ipynb
│   ├── inference.ipynb
├── requirements.txt
├── README.md
└── reports/
    └── project_report.pdf
```

---

## Quickstart
1. Upload `train(1)(train(1)).csv` and `test2(test(1)).csv` into project directory. Put all satellite images into `images_mapbox/` named as `{id}.png`.  
2. Open `notebooks/preprocessing.ipynb` and run cells top-to-bottom. This produces cleaned data in `data/processed/` and artifacts in `artifacts/`.  
3. Run `notebooks/feature_extraction.ipynb` to compute image features and ResNet embeddings (saves `artifacts/image_features_resnet.csv`).  
4. Run `notebooks/model_training.ipynb` to train models and save the best pipeline to `artifacts/best_model.pkl`.  
5. Run `notebooks/submission.ipynb` to predict and create submission CSV named `Piyush_Sagatani_23115104_Submission.csv`.  
6. Use `notebooks/inference.ipynb` to run single-point inference.

---

## Requirements
Install the environment:
```
pip install -r requirements.txt
```

Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `torchvision`, `opencv-python`, `joblib`, `matplotlib`, `seaborn`.

---

## Artifacts to expect after running notebooks
- `artifacts/image_features_resnet.csv` — image embeddings for ResNet18 (512 dims)
- `artifacts/test_image_features.csv`
- `artifacts/scaler.pkl`, `artifacts/num_imputer.pkl`, `artifacts/pca_numeric.pkl`, `artifacts/cat_imputer.pkl`, `artifacts/feature_names.pkl`, `artifacts/house_price_model.pkl`, `artifacts/kmeans.geo.pkl`, `artifacts/zip_rank.pkl`, `artifacts/zip_wealth_map.pkl`
- `artifacts/best_model.pkl` — saved pipeline for final predictions
- `Piyush_Sagatani_23115104_Submission.csv` — final submission file

---

## Contact & Attribution
**Author:** Piyush Sagatani