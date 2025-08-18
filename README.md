# House Price Prediction using Apache Spark MLlib

A machine learning project that predicts house prices using Apache Spark and MLlib. This project demonstrates end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, and evaluation.

## 🏠 Project Overview

This project uses the California Housing dataset to predict median house values using various features such as location coordinates, housing age, room counts, and ocean proximity. The implementation leverages Apache Spark's distributed computing capabilities for scalable machine learning.

## 📊 Dataset

The project uses `housing.csv` containing California housing data with the following features:

- **longitude**: Longitude coordinate of the house location
- **latitude**: Latitude coordinate of the house location
- **housing_median_age**: Median age of houses in the area
- **total_rooms**: Total number of rooms in the area
- **total_bedrooms**: Total number of bedrooms in the area
- **population**: Population of the area
- **households**: Number of households in the area
- **median_income**: Median income of households in the area
- **ocean_proximity**: Proximity to ocean (categorical: NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND)
- **median_house_value**: Target variable - median house value (what we're predicting)

## 🛠️ Technology Stack

- **Apache Spark**: Distributed computing framework
- **PySpark**: Python API for Spark
- **Spark MLlib**: Machine learning library
- **Java 21**: Required for Spark execution
- **Jupyter Notebook**: Interactive development environment

## 🔧 Setup & Installation

### Prerequisites

- Java 21 (OpenJDK recommended)
- Python 3.7+
- Apache Spark (installed via PySpark)

### Environment Setup

```bash
# Install required packages
pip install pyspark jupyter pandas

# Ensure Java 21 is installed
java -version
```

## 📝 Machine Learning Pipeline

### 1. Data Loading & Exploration

- Load housing data from CSV
- Explore data structure and statistics
- Add unique ID column for tracking

### 2. Data Preprocessing

- **Missing Value Handling**: Use `Imputer` to replace null values with mean
- **Feature Selection**: Separate numerical and categorical features
- **Train-Test Split**: 70% training, 30% testing

### 3. Feature Engineering

#### Numerical Features Processing:

- **Vector Assembly**: Combine numerical features into single vector
- **Standard Scaling**: Normalize features using StandardScaler
  ```
  scaled_value = (original_value - mean) / standard_deviation
  ```

#### Categorical Features Processing:

- **String Indexing**: Convert categorical strings to numerical indices
- **One-Hot Encoding**: Transform categorical variables to binary vectors
  - Prevents false ordering assumptions
  - Each category gets independent representation

#### Final Feature Assembly:

- Combine scaled numerical features and one-hot encoded categorical features
- Create final feature vector for model training

### 4. Model Training

- **Algorithm**: Linear Regression
- **Features**: `final_feature_vector` (combined numerical + categorical)
- **Target**: `median_house_value`
- Train on 70% of data, validate on remaining 30%

### 5. Model Evaluation

Comprehensive evaluation using multiple regression metrics:

- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target
- **Mean Absolute Error (MAE)**: Average absolute difference
- **R² Score**: Coefficient of determination (proportion of variance explained)

## 📁 Project Structure

```
spark_mllib/
├── spark.ipynb          # Main Jupyter notebook
├── housing.csv          # Dataset
├── README.md           # This file
└── venv/               # Python virtual environment (if used)
```

## 🚀 Usage

1. **Start Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Open `spark.ipynb`**

3. **Run cells sequentially** - the notebook is organized in logical steps:
   - Data loading and exploration
   - Preprocessing and feature engineering
   - Model training and evaluation

## 📈 Key Features

- **Scalable Processing**: Uses Spark for handling large datasets
- **Complete ML Pipeline**: End-to-end machine learning workflow
- **Feature Engineering**: Proper handling of both numerical and categorical data
- **Model Evaluation**: Comprehensive metrics for regression analysis
- **Production Ready**: Follows ML best practices (proper train/test split, feature scaling)

## 🔍 Results Interpretation

The model outputs several metrics to evaluate performance:

- **Lower MSE/RMSE/MAE**: Better prediction accuracy
- **Higher R²**: Better model fit (closer to 1.0 is better)
- **RMSE units**: In dollars (same as house prices), easy to interpret

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:

- Additional feature engineering techniques
- Different ML algorithms comparison
- Hyperparameter tuning
- Data visualization enhancements

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

_Built with ❤️ using Apache Spark and MLlib_
