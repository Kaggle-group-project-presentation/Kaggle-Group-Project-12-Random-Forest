
# Predicting Passenger Fate Using Machine Learning

A machine learning project for classifying passenger fate using the [Spaceship Titanic dataset](https://www.kaggle.com/competitions/spaceship-titanic/data). This project covers the full data science pipeline: data loading, preprocessing, analysis, feature engineering, model training, hyperparameter tuning, and predictions using Random Forests.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Description](#data-description)
- [Usage](#usage)
- [Approach](#approach)
- [Results](#results)
- [Contributors](#contributors)
- [References](#references)

---

## Project Structure

```
.
├── Kaggle Group Project Code.ipynb      # Main Jupyter Notebook for the workflow
├── train.csv          # Training dataset
├── test.csv           # Test dataset
├── sample_submission.csv # Sample submission format
├── README.md          # Project documentation (this file)
```

---

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install all requirements with:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Data Description

The project uses the Spaceship Titanic dataset, which contains passenger data with the goal of predicting whether each passenger was "Transported" (binary classification). The features include:

- **HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**

---

## Usage

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Arbabyounis46/Project_Breast_Cancer_Classification.git](https://github.com/Kaggle-group-project-presentation/Kaggle-Group-Project-12-Random-Forest
   ```

2. **Upload datasets to your working directory (e.g., Colab or local):**
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

3. **Open the notebook:**
   - Open `Kaggle Group Project Code.ipynb` in Jupyter Notebook or Google Colab.

4. **Run all cells:**
   - The notebook will walk you through data loading, preprocessing, visualization, modeling, and predictions.

---

## Approach

- **Data Preprocessing:**  
  - Drop unnecessary columns (`Name`, `PassengerId`)
  - Impute missing values with median (numeric) and mode (categorical)
  - Encode categorical variables
- **Exploratory Data Analysis (EDA):**
  - Visualize class balance, distributions, and relationships
- **Feature Engineering:**  
  - Feature importance determined via Random Forests
  - Select most important features for modeling
- **Model Selection:**  
  - Hyperparameter tuning using GridSearchCV on Random Forest Classifier
  - 5-fold cross-validation
- **Evaluation:**  
  - Report best parameters and cross-validated accuracy
  - Generate predictions on test set

---

## Results

- **Best Model:** Random Forest Classifier
- **Best Hyperparameters:**  
  - `max_depth`: 10  
  - `min_samples_leaf`: 2  
  - `min_samples_split`: 10  
  - `n_estimators`: 100
- **Best Cross-Validation Accuracy:** ~0.797
- **Output:** Submission file with `PassengerId` and predicted `Transported` values (True/False)

---

## Contributors

- Fatimah
- Ata
- Arbab
- Usman

---

## References

- [Spaceship Titanic Competition - Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [pandas documentation](https://pandas.pydata.org/)

---

*For questions or suggestions, please open an issue or contact the contributors directly.*

