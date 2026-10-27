<div align="center">

# Titanic Survival Prediction using Logistic Regression

**A beginner-friendly Machine Learning project with Python and scikit-learn**

Predict whether a Titanic passenger survived (`1`) or did not survive (`0`) using passenger information.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)

</div>

---

## 📌 Project Introduction

The Titanic dataset contains passenger details such as passenger class, sex, age, number of family members aboard, ticket fare, and port of embarkation. The goal is to learn patterns from historical records and predict the binary target **`Survived`**.

- `0` = Did not survive
- `1` = Survived

This is a **supervised binary classification** problem because the training dataset already contains the correct outcome for each passenger.

> **Important:** This model learns statistical patterns from historical data. It does not establish why any individual passenger survived, and its predictions should not be treated as causal explanations.

## 🖼️ Project workflow

![Titanic ML workflow](images/workflow.png)

## 🧠 Why Logistic Regression?

Despite the word *regression* in its name, **Logistic Regression is commonly used for classification**. It estimates the probability that an input belongs to a class, then converts that probability into a class label using a decision threshold.

For this project, the model estimates:

**P(Survived = 1 | passenger features)**

### How it works — conceptually

1. **Read the features:** the model receives values such as passenger class, sex, age, family counts, fare, and embarkation port.
2. **Combine the information:** it calculates a weighted score from the input features.
3. **Convert the score into a probability:** the sigmoid function maps the score to a value between 0 and 1.
4. **Make a class prediction:** with the usual `0.5` threshold, a probability of at least `0.5` is classified as `1`; otherwise, it is classified as `0`.

![Sigmoid curve used by Logistic Regression](images/sigmoid.png)

### The mathematics behind it

First, the model calculates a score:

\[
z = b + w_1x_1 + w_2x_2 + \dots + w_nx_n
\]

Then the sigmoid function converts that score into a probability:

\[
P(y=1) = \frac{1}{1 + e^{-z}}
\]

Where:

- `x` = passenger features
- `w` = learned feature weights
- `b` = bias/intercept
- `P(y=1)` = estimated probability of survival

The model learns the weights and intercept from the training data by minimizing a classification loss. The code uses scikit-learn's `LogisticRegression` implementation.

### A simple example

Imagine the model estimates a survival probability of `0.78` for a passenger.

- `0.78 ≥ 0.5`
- Predicted class = `1` (survived)

This is only an example to explain the threshold; it is **not** a prediction for a specific passenger in the dataset.

## 🔄 Project pipeline

| Stage | What happens | Why it matters |
|---|---|---|
| Data collection | Load the Titanic CSV file | Provides the historical records |
| Data inspection | Check rows, columns, types, and missing values | Helps understand data quality |
| Preprocessing | Remove the `Cabin` column, fill missing `Age` and `Embarked` values, encode categories | Makes the data usable by the model |
| Feature/target separation | Store passenger inputs in `X` and `Survived` in `Y` | Separates inputs from the answer |
| Train/test split | Use 80% for training and 20% for testing | Evaluates the model on unseen rows |
| Model training | Fit Logistic Regression on training data | Learns feature weights |
| Evaluation | Compare predictions with actual labels using accuracy | Measures the fraction of correct predictions |

## 🧹 Data preprocessing in this notebook

The notebook applies these transformations:

- Drops `Cabin` because it has many missing values in the supplied dataset.
- Fills missing `Age` values with the mean age.
- Fills missing `Embarked` values with the most frequent category (mode).
- Encodes `Sex`: `male → 0`, `female → 1`.
- Encodes `Embarked`: `S → 0`, `C → 1`, `Q → 2`.
- Drops `PassengerId`, `Name`, and `Ticket` from the model inputs.
- Uses `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked` as features.

**Preprocessing note:** the notebook performs these fills and encodings before the train/test split. For a stricter production-quality evaluation, fit preprocessing rules on the training set only (for example, with a scikit-learn `Pipeline`) to avoid data leakage. Also, treating port labels as `0`, `1`, and `2` introduces an artificial numeric order; one-hot encoding is often a better option for nominal categories.

## 🧪 Train/test split

The notebook uses:

```python
train_test_split(X, Y, test_size=0.2, random_state=2)
```

- `test_size=0.2` reserves 20% of rows for testing.
- `random_state=2` makes the split reproducible.
- In the notebook, the dataset has **891 rows** and the split produces **712 training rows** and **179 testing rows**.

The model should be judged primarily on test data, because training accuracy measures performance on records the model has already seen.

## ⚙️ Training the Logistic Regression model

The core model steps are:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

- `LogisticRegression(...)` creates the classifier.
- `max_iter=1000` gives the optimization algorithm more iterations to converge than the default in the notebook, which showed a convergence warning.
- `fit(X_train, Y_train)` learns the weights from the training examples.

## 📊 Results

The notebook reported the following accuracy values:

| Evaluation set | Accuracy |
|---|---:|
| Training data | **80.76%** |
| Test data | **78.21%** |

Accuracy is the percentage of predictions that match the true labels. The test result means the model correctly classified about 78 out of every 100 test records in this particular split.

**Accuracy is not the whole story.** A fuller evaluation should also inspect a confusion matrix, precision, recall, and F1-score. The notebook's reported figures are results from its recorded run; they may vary if preprocessing or model settings change.

## 🛠️ Technologies used

- **Python** — programming language
- **NumPy** — numerical operations
- **Pandas** — data loading and manipulation
- **Matplotlib** — plotting
- **Seaborn** — statistical visualizations
- **scikit-learn** — train/test splitting, Logistic Regression, and accuracy calculation

## ▶️ How to run

### Option 1: Google Colab

1. Open the notebook linked in this repository.
2. Upload the Titanic training CSV file when prompted, or mount the dataset in Colab.
3. Update the CSV path in `pd.read_csv(...)` if the filename or location is different.
4. Run the notebook cells from top to bottom.

### Option 2: Run locally

1. Install Python 3.
2. Install the libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. Place the Titanic CSV file in your project folder.
4. Change the notebook's CSV path to the local file path.
5. Open and run the `.ipynb` notebook in Jupyter Notebook or VS Code.

## 📁 Suggested repository structure

```text
titanic-survival-prediction/
├── README.md
├── Titanic_survival_prediction_with_python_using_ML.ipynb
├── train.csv
└── images/
    ├── workflow.png
    └── sigmoid.png
```

Keep the dataset in the repository only if its licence and repository policies allow it. Otherwise, explain where to obtain it.

## 🚀 Possible improvements

- Use a `Pipeline` and `ColumnTransformer` so preprocessing is learned only from training data.
- Use one-hot encoding for categorical features such as `Embarked` and `Pclass`.
- Add a confusion matrix, precision, recall, and F1-score.
- Compare Logistic Regression with Decision Tree, Random Forest, and other classifiers.
- Tune the decision threshold based on the evaluation goal.
- Use cross-validation to estimate performance more robustly.

## 🎯 Learning outcomes

By completing this project, a beginner can practise:

- Understanding a tabular dataset
- Handling missing data and encoding categorical values
- Separating features from the target
- Splitting data into training and testing sets
- Training a Logistic Regression classifier
- Understanding probability thresholds and binary predictions
- Evaluating a model with accuracy and identifying further evaluation needs

## 👤 Author

**Mohd Arhan**

*Machine Learning • Python • Data Analysis*

---

<div align="center">

If this project helped you understand Logistic Regression, consider ⭐ starring the repository.

</div>
