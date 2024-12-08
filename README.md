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
Maintaining a diet can be beneficial for many reasons, such as nutritional restrictions, bodybuilding, religious guidelines, and many more. Therefore, knowing what type of diet a meal is is very important to sticking to your goals/needs. However, with so many ingredients in foods nowadays, it’s very hard to tell if the ingredients follow your diet guidelines. This is where our model, “Diet Warriors”, comes in. Our model aims to classify diet types based on nutrition to make sure you’re sticking to your dietary restrictions as best as possible.
The model was chosen because it helps solve a growing need for dietary assistance in a time where heavily processed and complex meals are common. By having a good predictive model, we can help reduce the risk of health issues arising from consuming restricted ingredients, spur physical health goals through making sure meals fit your diet type, and help people adhere to their religious dietary restrictions.

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

For preprocessing, we first begin by improving the dataset's readability by renaming columns to more concise forms, converting column names like `Diet_type` to `Diet` and `Recipe_name` to `Recipe`.

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
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=23)
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
Validation Accuracy: 0.5292

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
Validation Accuracy: 0.5733

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
Validation Accuracy: 0.7462

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

In Milestone 3, we began by training a Logistic Regression Classifier on our diet classification task. As shown above, the model demonstrates relatively consistent performance across all three datasets, with accuracies of 54% on the training and test sets, and 53% on the validation data. The small difference between training and test accuracy (only about 0.74 percentage points) suggests that overfitting is not a significant issue. However, the overall low accuracy across all splits indicates that the model is likely underfitting, meaning it struggles to capture the underlying patterns in the relationship between our features (nutritional content and cuisine type) and the target (diet types).

After some discussion, we hypothesized that the relatively low accuracy with the Logistic Regression model suggests several potential issues. First, we considered the possibility that the dataset itself might not have enough distinguishing features. Specifically, there may be overlap in the nutritional profiles across different diet types. For example, in the pie chart above, we observe that the overall distribution of fat, protein, and carbs in the ‘mediterranean’ and ‘dash’ diets are relatively similar, differing only slightly. Another limitation of our initial model is that the relationship between nutritional values and diet types may be more non-linear than our model can capture. Since Logistic Regression is a linear model, it is not able to account for more complex relationships that could exist between the features and the target.

After analyzing the results of our first model, we decided to train our second model using a Support Vector Machine (SVM). For this model, we chose the Radial Basis Function (RBF) kernel because it transforms the data into a higher-dimensional space, allowing the SVM to find a non-linear decision boundary. This choice was made to test our hypothesis from the previous model's analysis and to explore whether a non-linear model would improve performance. We also tried to tune the hyperparameters such as regularization strength (C) using GridSearchCV to see if the accuracy would improve for our SVM. Specifically, we tested multiple values for C (0.1,1,10) and γ (0.1,0.01,0.001) while keeping the kernel fixed.

After running the model, we observed an improvement of about 3-4 percentage points across all datasets, which supports our hypothesis that the dataset contains more complexity than initially anticipated. However, despite the improvement, the overall accuracy remains relatively modest. Even after hypertuning, the accuracy still remained consistent to the previous ones. This suggests that, while the SVM with the RBF kernel is better suited for capturing non-linear relationships, it still struggles to fully capture the complexity of diet classification with our current features.

For our final model, we chose to train a Gradient Boosting Classifier, which can naturally handle feature interactions through its tree-based architecture. We believed that this was particularly relevant for our dataset, where the relationship between nutritional content (protein, carbs, fat) and diet types might involve complex interdependencies. For instance, the combination of high fat and low carbs may be more indicative of a keto diet than either feature alone. The sequential nature of Gradient Boosting, where each tree corrects the errors of the previous one, allows the model to learn intricate patterns gradually. This is especially useful when diet types have subtle distinctions in their nutritional profiles that simpler models might miss. Unlike SVM, which uses a single kernel function, Gradient Boosting can automatically learn feature transformations through its tree structure, making it well-suited to handle both numerical features (nutritional values) and one-hot encoded categorical features (cuisine types) in a unified way.

For our hyperparameters, we chose `n_estimators=100` to ensure the model has enough boosting stages to capture patterns without being overly computationally expensive. The `learning_rate=0.1` balances the contribution of each tree, providing steady and controlled learning while minimizing the risk of overfitting. The `max_depth=5` limits the complexity of individual trees, enabling the model to capture moderately complex relationships without overfitting to the training data.

The result of our final model, as shown in the above section, demonstrated a significant increase in both training and validation accuracies, but the test accuracy only showed a marginal improvement over the SVM's performance, which was unexpected. The large gap between training/validation and test performance (approximately 19 percentage points) indicates that the model has significant overfitting issues. Our team analyzed this overfitting and identified several possible reasons:

First, the hyperparameters we chose—such as the relatively high depth and the large number of estimators—might not be suitable for the complexity of our dataset. These parameters could make the model overly complex, leading it to learn too closely from the training data rather than generalizing to unseen data. Second, while we observed some complexity in our data earlier, the model's architecture may assume a higher level of intricacy than necessary. The choice of deeper trees and a larger number of estimators could be capturing noise in the data rather than the true patterns, leading to overfitting. Finally, there is a possibility that the dataset contains noise or irrelevant features, which the model could be learning instead of the actual underlying structure. This is particularly a concern with Gradient Boosting models, as they iteratively focus on correcting residuals, which can result in overfitting to noise if not properly regularized.

## Conclusion
In this project, we explored the possibility of classifying different recipes solely from their nutritional facts. Despite the use of three different models, we only saw minor increases in performance (measured in accuracy, precision, and recall). We believed that this was due to the extreme overlap between recipes of different diet types shown in our data exploration pairplot. The features used to train the model were quite limited to only the carbs, protein, and fats of the recipe. Because of the limited features, there were not enough dimensions for our models to effectively separate the data into different classes despite the more complex models used. This led our models to severely underfit the data unable to effectively classify each recipe. 

For future works, a starting point would be to enhance our dataset with more nutritional facts of each recipe. Instead of just the 3 features (Fat, Carbs, and Protein), we could expand it to the full nutritional list including sodium levels, minerals, expanding fats to Saturated versus Trans, Dietary Fiber, etc. This would involve a lot more data collection since the base dataset we used only have the 3 features mentioned earlier. So for future work, we should spend more time on data collection. We could then move on to utilizing more complex models with better feature extraction such as ResNet or DenseNet because our base 3 features have a lot of overlap in impact. This should allow for a more robust model. 

## Statement of Collaboration

Skyler Goh: Contributed to write-up, writing code for preprocessing, model metrics, and result graphs. Helped contribute to group discussion, feedback, and analysis. 

Kevin Yan: Set up project, contributed to write-up, programmed, trained, and tuned xgboost model. Helped contribute to group discussion, feedback, and analysis. 

Phillip Wu: Contributed to write-up, programmed, trained, and tuned gradient boost model. Helped contribute to group discussion, feedback, and analysis. 

Luffy Saito: Contributed to write-up, programmed pre-processing,  data-encoding, data-splitting, and first logistic model, programmed and trained svm model, 10x bug fixer, project organizer, Helped contribute to group discussion, feedback, and analysis. 


