# Titanic Passenger Survival Prediction

## Dataset Description
## Overview
The data has been split into two groups:
- training set (train.csv)
- test set (test.csv)
**The training set** should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

**The test set** should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

### **Data Dictionary**
Below is a description of the variables in the dataset:

| Variable   | Definition                                 | Key                                      |
|------------|--------------------------------------------|------------------------------------------|
| `survival` | Survival                                   | 0 = No, 1 = Yes                          |
| `pclass`   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| `sex`      | Sex                                        |                                          |
| `age`      | Age in years                               |                                          |
| `sibsp`    | # of siblings/spouses aboard the Titanic   |                                          |
| `parch`    | # of parents/children aboard the Titanic   |                                          |
| `ticket`   | Ticket number                              |                                          |
| `fare`     | Passenger fare                             |                                          |
| `cabin`    | Cabin number                               |                                          |
| `embarked` | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

### **Variable Notes**
- **`pclass`**: A proxy for socio-economic status (SES):
  - 1st = Upper
  - 2nd = Middle
  - 3rd = Lower
- **`age`**: Age is fractional if less than 1. If the age is estimated, it is in the form of xx.5.
- **`sibsp`**: The dataset defines family relations as:
  - Sibling = brother, sister, stepbrother, stepsister
  - Spouse = husband, wife (mistresses and fiancés were ignored)
- **`parch`**: The dataset defines family relations as:
  - Parent = mother, father
  - Child = daughter, son, stepdaughter, stepson
  - Some children traveled only with a nanny, therefore `parch=0` for them.


## Project Overview
This project involves predicting the survival of passengers on the Titanic using the Titanic dataset. The analysis includes data visualization, data preprocessing, and building machine learning models to make accurate predictions. The models utilized in this project include `RandomForestClassifier`, `XGBClassifier`, and `StackingClassifier`, along with hyperparameter tuning techniques such as `RandomizedSearchCV` and `GridSearchCV`.

## Dataset
The dataset contains information about Titanic passengers, including features like age, gender, ticket class, and whether they survived. The goal is to use these features to predict whether a passenger survived or not.

## Key Features of the Project
1. **Data Visualization**: 
   - Exploratory Data Analysis (EDA) was conducted using Matplotlib and Seaborn to visualize the relationships between features and the target variable (`Survived`).
   - Key visualizations include distributions of passengers by age, gender, class, and survival rate.
   
2. **Data Preprocessing**:
   - Handling missing data (e.g., imputing missing age and embarked values).
   - Encoding categorical features such as `Sex`, `Embarked`, and `Pclass`.
   - Feature scaling and engineering, including creating new features based on existing data (e.g., `FamilySize`).

3. **Model Building**:
   - **RandomForestClassifier**: A robust and versatile ensemble model that was trained on the processed dataset.
   - **XGBClassifier**: A powerful gradient boosting model was employed to improve prediction accuracy.
   - **StackingClassifier**: A meta-model that combines the predictions of RandomForest and XGBoost to improve overall performance.

4. **Hyperparameter Tuning**:
   - **RandomizedSearchCV**: Used to quickly explore a wide range of hyperparameters for `RandomForestClassifier` and `XGBClassifier`.
   - **GridSearchCV**: Used for fine-tuning hyperparameters once promising ranges were identified using RandomizedSearch.

## Project Structure
The project files are structured as follows:
- `data/`: Contains the Titanic dataset used for analysis.
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, and model training.
- `README.md`: Project documentation.

## Results and Performance
The models were evaluated based on accuracy, precision, recall, and F1-score. After tuning the hyperparameters, the **StackingClassifier** provided the best overall performance. Key highlights:
- **RandomForestClassifier**: Achieved good baseline performance with default parameters.
- **XGBClassifier**: Showed significant improvement in precision and recall.
- **StackingClassifier**: Combined the strengths of RandomForest and XGBoost to improve overall accuracy.

## Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-Learn**: For machine learning model building, including RandomForestClassifier, StackingClassifier, and hyperparameter tuning (RandomizedSearchCV, GridSearchCV).
- **XGBoost**: For implementing XGBClassifier.
