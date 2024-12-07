# UCSD-CSE-151A-Project

#### [Click here for GitHub Repo](https://github.com/DaikonPlays/diet-warriors/tree/main)

## Team Members
| Name | GitHub |
|------|--------|
| Kevin Yan   | [https://github.com/DaikonPlays](https://github.com/DaikonPlays)  |
| Luffy Saito | [https://github.com/rsa1to](https://github.com/rsa1to) |
| Skyler Goh  | [https://github.com/SkylerGoh](https://github.com/SkylerGoh) |
| Phillip Wu  | [https://github.com/philliptwu](https://github.com/philliptwu) |

## Introduction

## Figures

## Methods

### **Data Exploration:**

The first step in our data exploration was to identify the various attributes present in our dataset. Here are our findings:

| Attribute | Explanation |
|-----------|-------------|
|**Diet_type** | The type of diet the recipe fits into |
| **Recipe_name**| The name of the recipe|
| **Cuisine_type**| The type of cuisine the recipe belongs to |
| **Protein(g)**| The amount of protein in the recipe, measured in grams |
| **Carbs(g)**| The amount of carbohydrates in the recipe, measured in grams |
| **Fat(g)**| The amount of fat in the recipe, measured in grams |
| **Extraction_day**| The day the data was extracted |
| **Extraction_time**| The time the data was extracted |

Our target variable is `Diet_Type` as we aim to predict which diet types best suit our users. The class contains the following unique labels:

```
diet_types = diet_data['Diet_type'].unique()
print(diet_types)
-----------------------------------------------
['paleo' 'vegan' 'keto' 'mediterranean' 'dash']
```

We also examined the unique labels in the `Cuisine_Type` attribute:
```
cuisine_types = diet_data['Cuisine_type'].unique()
print(cuisine_types)
------------------------------------------------------------------------
['american' 'south east asian' 'mexican' 'chinese' 'mediterranean'
 'italian' 'french' 'indian' 'nordic' 'eastern europe' 'central europe'
 'kosher' 'british' 'caribbean' 'south american' 'middle eastern' 'asian'
 'japanese' 'world']
```

In order to ensure data quality and avoid any errors, we are going to see if there are any null values in any of our columns, drop them if we do:
```
null_counts = diet_data.isnull().sum()
print(null_counts)
------------------------------------------
Diet_type          0
Recipe_name        0
Cuisine_type       0
Protein(g)         0
Carbs(g)           0
Fat(g)             0
Extraction_day     0
Extraction_time    0
dtype: int64
```
We see that there are no null values. YAY!

To visualize relationships in our data, we used two approaches:
1. The `pairplot` function from seaborn to visualize pairwise relationships between our numerical features (Protein, Fat, Carbs)
2. Pie charts to display the proportional composition of **Protein(g)**, **Carbs(g)**, and **Fat(g)** across different diet types in the diet_data dataset

Below are our visualizations:

![Pair Plot](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/diet_type_pair_plot.png)

![Pie Chart](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/diet_type_pie_chart.png)

### **Pre-Processing:**

For preprocessing, we firs tbegin by improving the dataset's readability by renaming columns to more concise forms, converting column names like `Diet_type` to `Diet` and `Recipe_name` to `Recipe`.

```
diet_data = diet_data.rename(
    columns={
        'Diet_type': 'Diet',
        'Recipe_name': 'Recipe',
        'Cuisine_type': 'Cuisine',
        'Protein(g)': 'Protein',
        'Carbs(g)': 'Carbs',
        'Fat(g)': 'Fat',
        })
```

We then cleaned the dataset by removing unnecessary temporal information, specifically the  `Extraction_day` and `Extraction_time` columns, which were not relevant to our classification task.

```
dd_processed = diet_data.drop(columns=['Extraction_day', 'Extraction_time'])
```

The feature engineering phase involved transforming categorical variables into a format suitable for machine learning algorithms. We applied Label Encoding to our target variable (`Diet`), and implemented One-Hot Encoding for the `Cuisine` categorical feature using pandas' `get_dummies()` function, which created binary columns for each cuisine type.

```
encoder = LabelEncoder()

# Encode Diet Type & Cuisine Type
dd_processed['Diet'] = encoder.fit_transform(dd_processed['Diet'])
dd_processed = pd.get_dummies(dd_processed, columns=['Cuisine'])
```

Finally, we will split our data in features and taget sets. We will use a 80-20 ratio for taining-test split, and will also create validation sets for model development and tuning. This validation split would later allow us to assess our model's performance during the development phase without compromising the integrity of our final test set.

```
X = dd_processed.drop(columns=['Recipe','Diet'])
y = dd_processed['Diet']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# validation test
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

### **Model 1:**

We chose our first model to be a simple Logistic Regression classifier. After training, we generated predictions on three different datasets: the training set, test set, and validation set. We calculated accuracy scores for each of these predictions using the `accuracy_score` metric, storing these scores in variables for training, test, and validation accuracy respectively. The resulting performance of the model is noted in the section below. 

```
# We are starting with a simple Logistic Regression to get a baseline performance
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predictions
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
y_pred_val = logreg.predict(X_test_val)

logreg_training_accuracy = accuracy_score(y_train, y_pred_train)
logreg_test_accuracy = accuracy_score(y_test, y_pred_test)
logreg_validation_accuracy = accuracy_score(y_test_val, y_pred_val)
```
### **Model 2:**

Similarly to our Logistic Regression implementation, we trained a Support Vector Machine (SVM) classifier using a radial basis function (RBF) kernel.

```
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

# Predictions
y_pred_train = svc.predict(X_train)
y_pred_test = svc.predict(X_test)
y_pred_val = svc.predict(X_test_val)

svc_training_accuracy = accuracy_score(y_train, y_pred_train)
svc_test_accuracy = accuracy_score(y_test, y_pred_test)
svc_validation_accuracy = accuracy_score(y_test_val, y_pred_val)
```

### **Model 3:**

Finally, we trained a Gradient Boosting Classifier in the same manner as the model above. This model was configured with the hyperparameters: 100 estimators, a learning rate of 0.1, maximum tree depth of 5, and a validation fraction of 0.1. 

```
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, validation_fraction=0.1, random_state=42)
gb_model.fit(X_train, y_train)

gb_model.score(X_test, y_test)
gb_model.score(X_test_val, y_test_val)

# Predictions
y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)
y_pred_val = gb_model.predict(X_test_val)

gb_training_accuracy = accuracy_score(y_train, y_pred_train)
gb_test_accuracy = accuracy_score(y_test, y_pred_test)
gb_validation_accuracy = accuracy_score(y_test_val, y_pred_val)
```

## Results

Below is our calculated accuracy scores and a classification report for the Logistic Regression classifier:

### **Model 1**

```
Training Accuracy: 0.5452
Test Accuracy: 0.5378
Validation Accuracy: 0.5388

Classification Report:
               precision    recall  f1-score   support

           0       0.44      0.32      0.37       348
           1       0.49      0.79      0.61       303
           2       0.71      0.75      0.73       361
           3       0.34      0.11      0.16       253
           4       0.52      0.64      0.58       297

    accuracy                           0.54      1562
   macro avg       0.50      0.52      0.49      1562
weighted avg       0.51      0.54      0.51      1562

```
To visualize our model's performance, we created a fitting graph comparing accuracies across different data splits:

![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/logreg_fitting_graph.png)

Further, we constructed a confusion matrix to analyze the model's performance across different diet types. For each diet category, we calculated and displayed TP, FN, TN, FN.

![Confusion Matrix](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/logreg_cm.png)
```
Class  dash
True Positives:  113
False Positives:  142
True Negative:  1072
False Negative:  235

Class  keto
True Positives:  238
False Positives:  245
True Negative:  1014
False Negative:  65

Class  mediterranean
True Positives:  272
False Positives:  110
True Negative:  1091
False Negative:  89

Class  paleo
True Positives:  27
False Positives:  53
True Negative:  1256
False Negative:  226

Class  vegan
True Positives:  190
False Positives:  172
True Negative:  1093
False Negative:  107
```

### **Model 2**

Our results and figures for the SVM model:

```
Training Accuracy: 0.5852
Test Accuracy: 0.5589
Validation Accuracy: 0.5701

Classification Report:
               precision    recall  f1-score   support

           0       0.49      0.39      0.43       348
           1       0.56      0.73      0.63       303
           2       0.70      0.77      0.73       361
           3       0.33      0.19      0.24       253
           4       0.55      0.65      0.60       297

    accuracy                           0.56      1562
   macro avg       0.53      0.54      0.53      1562
weighted avg       0.54      0.56      0.54      1562
```

![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/svc_fitting_graph.png)
![Confusion Matrix](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/svc_cm.png)

```
Class  dash
True Positives:  135
False Positives:  141
True Negative:  1073
False Negative:  213

Class  keto
True Positives:  220
False Positives:  174
True Negative:  1085
False Negative:  83

Class  mediterranean
True Positives:  277
False Positives:  118
True Negative:  1083
False Negative:  84

Class  paleo
True Positives:  48
False Positives:  99
True Negative:  1210
False Negative:  205

Class  vegan
True Positives:  193
False Positives:  157
True Negative:  1108
False Negative:  104
```

### **Model 3**

Finally, our results and figures for the Gradient Boost Classifier:

```
Training Accuracy: 0.7534
Test Accuracy: 0.5653
Validation Accuracy: 0.7454

Classification Report:
               precision    recall  f1-score   support

           0       0.52      0.45      0.48       348
           1       0.58      0.70      0.63       303
           2       0.70      0.76      0.73       361
           3       0.36      0.26      0.30       253
           4       0.54      0.58      0.56       297

    accuracy                           0.57      1562
   macro avg       0.54      0.55      0.54      1562
weighted avg       0.55      0.57      0.55      1562
```
![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/gb_fitting_graph.png)
![Confusion Matrix](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/gb_cm.png)

```
Class  dash
True Positives:  157
False Positives:  144
True Negative:  1070
False Negative:  191

Class  keto
True Positives:  213
False Positives:  157
True Negative:  1102
False Negative:  90

Class  mediterranean
True Positives:  276
False Positives:  117
True Negative:  1084
False Negative:  85

Class  paleo
True Positives:  66
False Positives:  115
True Negative:  1194
False Negative:  187

Class  vegan
True Positives:  171
False Positives:  146
True Negative:  1119
False Negative:  126
```
## Discussion

## Conclusion

## Statement of Collaboration