# Heart Disease Prediction - ML Analysis Platform

A comprehensive machine learning platform for heart disease prediction using both supervised and unsupervised learning algorithms, with Supabase database integration and modern React frontend.

## Features

### Machine Learning Models

**Supervised Learning:**
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)

**Unsupervised Learning:**
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Fuzzy C-Means
- Gaussian Mixture Models (GMM)
- Autoencoders

### Platform Capabilities

- **Data Management**: Upload heart disease datasets via CSV
- **Model Training**: Train multiple ML models simultaneously
- **Predictions**: Make real-time predictions on new patient data
- **Comparison**: Comprehensive model performance analysis
- **Persistence**: All data stored in Supabase database
- **Visualizations**: Interactive charts and performance metrics

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node Dependencies

```bash
npm install
```

## Running the Application

### Start the Backend API

```bash
python app.py
```

The API will run on `http://localhost:5000`

### Start the Frontend (in a new terminal)

```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage Guide

### 1. Upload Your Dataset

1. Navigate to **Upload Data** page
2. Prepare your CSV file with these columns:
   - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`
   - `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`
3. Click "Choose File" and select your CSV
4. Click "Upload Dataset"

### 2. Train Models

1. Go to **Train Models** page
2. Train Supervised or Unsupervised models
3. View performance metrics and results

### 3. Make Predictions

1. Navigate to **Prediction** page
2. Select model and enter patient data
3. Get instant predictions

### 4. Compare Models

1. Go to **Comparison** page
2. View comprehensive performance comparison
3. See interactive charts and rankings

## Technology Stack

### Backend
- Python, Flask, scikit-learn, TensorFlow, Supabase

### Frontend
- React, Vite, React Router, Recharts


