# AI Medical Diagnosis System

A comprehensive machine learning-based health risk assessment system that predicts the risk of various diseases based on user health data. Built with Flask backend, modern frontend, and RandomForest machine learning model.

## 🚀 Features

- **AI-Powered Risk Assessment**: Predicts risk levels for 6 different diseases:
  - Heart Disease
  - Skin Cancer
  - Other Cancer
  - Depression
  - Diabetes
  - Arthritis

- **Modern Web Interface**: Beautiful, responsive UI with smooth animations
- **Real-time Predictions**: Fast API responses with detailed risk analysis
- **Comprehensive Health Data Analysis**: Takes into account:
  - Personal demographics (age, gender)
  - Physical measurements (height, weight, BMI)
  - Lifestyle factors (exercise, smoking, alcohol)
  - Dietary habits (fruit, vegetables, fried food consumption)
  - Medical history (check-ups)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdnanJuwle/AI-Medical-Diagnosis-System.git
   cd AI-Medical-Diagnosis-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install Flask flask-cors scikit-learn pandas numpy joblib
   ```

## 🏃 Running the Application

### 1. Start the Backend Server

```bash
python backend/app.py
```

The Flask server will start on `http://127.0.0.1:5000`

### 2. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a local server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8000
```

Then navigate to `http://localhost:8000` in your browser.

## 🎯 Usage

1. Fill out the health information form with your data
2. Click "Calculate BMI" if you need to compute your BMI from height and weight
3. Click "Analyze Health Risks" to get predictions
4. Review the risk assessment results for each disease

## 📊 Model Training

The system includes a comprehensive training script to train and evaluate the model:

```bash
python training/train_model.py
```

This script will:
- Load and preprocess the CVD dataset
- Train a RandomForest-based MultiOutputClassifier
- Evaluate model performance with detailed metrics
- Save the trained model and preprocessor to the `models/` directory
- Generate training metadata with accuracy scores

### Model Details

- **Algorithm**: MultiOutputClassifier with RandomForestClassifier
- **Features**: 36 input features (after preprocessing)
- **Targets**: 6 disease risk predictions
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Class Balancing**: Uses `class_weight='balanced'` to handle imbalanced data

## 📁 Project Structure

```
AI-Medical-Diagnosis-System/
├── backend/
│   └── app.py              # Flask API server
├── frontend/
│   ├── index.html          # Main web interface
│   ├── script.js           # Frontend JavaScript
│   └── styles.css          # Modern CSS styling
├── training/
│   └── train_model.py      # Model training script
├── models/
│   ├── medical_ai_model.pkl      # Trained ML model
│   ├── preprocessor_36.pkl       # Data preprocessor
│   └── training_metadata.json    # Training metrics and info
├── datasets/
│   └── CVD_cleaned.csv     # Training dataset
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔌 API Endpoint

### POST `/predict`

Predicts disease risk based on health data.

**Request Body:**
```json
{
  "age": "25-29",
  "gender": "Male",
  "height": 175,
  "weight": 70,
  "bmi": 22.9,
  "exercise": "Yes",
  "checkup": "Within the past year",
  "smoking_history": "No",
  "alcohol_consumption": "No",
  "fruit_consumption": 2,
  "green_vegetables_consumption": 3,
  "fried_food_consumption": 1
}
```

**Response:**
```json
{
  "Heart_Disease": 0,
  "Skin_Cancer": 0,
  "Other_Cancer": 0,
  "Depression": 0,
  "Diabetes": 0,
  "Arthritis": 0
}
```

Where `1` indicates high risk and `0` indicates low risk.

## 🎨 UI Features

- **Modern Design**: Clean, professional interface with gradient backgrounds
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile
- **Smooth Animations**: Engaging transitions and hover effects
- **Risk Summary**: Quick overview of high-risk conditions detected
- **Visual Indicators**: Color-coded risk levels (red for high risk, green for low risk)
- **Form Validation**: Real-time input validation and helpful tooltips

## 🔧 Technical Details

- **Backend**: Flask with CORS support
- **Frontend**: Vanilla JavaScript with modern CSS
- **ML Framework**: scikit-learn
- **Model**: RandomForestClassifier with MultiOutputClassifier
- **Preprocessing**: StandardScaler + OneHotEncoder
- **Features**: 36 processed features from 13 input features
- **Prediction Method**: Uses probability thresholds (0.25) for better sensitivity

## 📈 Model Performance

The model is trained on a dataset of 308,854 samples with the following characteristics:
- **Training Set**: 247,083 samples (80%)
- **Test Set**: 61,771 samples (20%)
- **Overall Accuracy**: ~64% (average across all diseases)
- **Per-Disease Accuracy**: Varies by disease (42-73%)

Note: Accuracy varies by disease due to class imbalance. The model uses balanced class weights to improve detection of high-risk cases.

## ⚠️ Important Medical Disclaimer

**This system is for educational and research purposes only.** 

It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. The predictions are based on statistical patterns in training data and may not accurately reflect individual health conditions.

## 🐛 Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Make sure the Flask backend server is running on `http://127.0.0.1:5000`
   - Check that no firewall is blocking the connection

2. **"Model not found" error**
   - Ensure the model files exist in the `models/` directory
   - Run the training script to generate model files if missing

3. **CORS errors**
   - Make sure `flask-cors` is installed
   - Verify CORS is enabled in `backend/app.py`

4. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

## 🔄 Recent Updates

- ✅ Added comprehensive training script (`training/train_model.py`)
- ✅ Upgraded to RandomForestClassifier for better performance
- ✅ Implemented probability-based predictions with adjustable thresholds
- ✅ Complete UI redesign with modern, responsive design
- ✅ Added training metadata and model documentation
- ✅ Improved prediction accuracy for high-risk cases
- ✅ Enhanced error handling and user feedback

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Adnan Juwle

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

**Remember**: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
