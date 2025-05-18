# Diabetes Prediction System

A machine learning project that predicts diabetes using Logistic Regression. This project demonstrates end-to-end implementation of a machine learning pipeline from data preprocessing to model evaluation.

## Project Structure
```
diabeteslr/
├── data/               # Dataset directory
├── plots/             # Generated visualizations
├── src/               # Source code
│   ├── data_preprocessing.py
│   ├── diabetes_model.py
│   └── main.py
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Features
- Custom Logistic Regression implementation
- Comprehensive data preprocessing pipeline
- L2 regularization and momentum optimization
- Early stopping mechanism
- Multiple evaluation metrics and visualizations

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabeteslr.git
cd diabeteslr
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python src/main.py
```

## Model Performance
- Accuracy: 72.73%
- Precision: 66.00%
- Recall: 56.90%

## Technical Details
- Custom Logistic Regression implementation with:
  - L2 regularization
  - Momentum optimization
  - Early stopping
  - Xavier/Glorot initialization
- Comprehensive data preprocessing:
  - Missing value handling
  - Feature scaling
  - Train-validation-test split
- Multiple evaluation metrics and visualizations:
  - Cost history
  - Confusion matrix
  - ROC curve
  - Feature correlations

## License
MIT License

## Author
[Your Name] 