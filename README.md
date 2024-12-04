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

| Attribute | Explanation |
|-----------|-------------|
|**Diet_type** | The type of diet the recipe fits into. |
| **Recipe_name**| The name of the recipe.|
| **Cuisine_type**| The type of cuisine the recipe belongs to. |
| **Protein(g)**| The amount of protein in the recipe, measured in grams. |
| **Carbs(g)**| The amount of carbohydrates in the recipe, measured in grams. |
| **Fat(g)**| The amount of fat in the recipe, measured in grams. |
| **Extraction_day**| The day the data was extracted. |
| **Extraction_time**| The time the data was extracted. |

```
diet_types = diet_data['Diet_type'].unique()
print(diet_types)
-----------------------------------------------
['paleo' 'vegan' 'keto' 'mediterranean' 'dash']
```
![Pair Plot](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/diet_type_pair_plot.png)

![Pie Chart](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/diet_type_pie_chart.png)

### **Pre-Processing:**

```
encoder = LabelEncoder()

# Encode Diet Type & Cuisine Type
dd_processed['Diet'] = encoder.fit_transform(dd_processed['Diet'])
dd_processed = pd.get_dummies(dd_processed, columns=['Cuisine'])
```

### **Model 1:**

```
# We are starting with a simple Logistic Regression to get a baseline performance
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```


### **Model 2:**

```
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
```

### **Model 3:**
```
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, validation_fraction=0.1, random_state=42)
gb_model.fit(X_train, y_train)
```

## Results

### **Model 1**

```
logreg_training_accuracy = accuracy_score(y_train, y_pred_train)
logreg_test_accuracy = accuracy_score(y_test, y_pred_test)
logreg_validation_accuracy = accuracy_score(y_test_val, y_pred_val)

print(f"Training Accuracy: {logreg_training_accuracy:.4f}")
print(f"Test Accuracy: {logreg_test_accuracy:.4f}")
print(f"Validation Accuracy: {logreg_validation_accuracy:.4f}")
------------------------------------------------------------
Training Accuracy: 0.5471
Test Accuracy: 0.5301
Validation Accuracy: 0.5500
```
![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/logreg_fitting_graph.png)
![Confusion Matrix](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/logreg_cm.png)


### **Model 2**

```
svc_training_accuracy = accuracy_score(y_train, y_pred_train)
svc_test_accuracy = accuracy_score(y_test, y_pred_test)
svc_validation_accuracy = accuracy_score(y_test_val, y_pred_val)

print(f"Training Accuracy: {svc_training_accuracy:.4f}")
print(f"Test Accuracy: {svc_test_accuracy:.4f}")
print(f"Validation Accuracy: {svc_validation_accuracy:.4f}")
------------------------------------------------------------
Training Accuracy: 0.5855
Test Accuracy: 0.5461
Validation Accuracy: 0.5885
```
![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/svc_fitting_graph.png)
![Confusion Matrix](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/svc_cm.png)

### **Model 3**
```
gb_training_accuracy = accuracy_score(y_train, y_pred_train)
gb_test_accuracy = accuracy_score(y_test, y_pred_test)
gb_validation_accuracy = accuracy_score(y_test_val, y_pred_val)

print(f"Training Accuracy: {gb_training_accuracy:.4f}")
print(f"Test Accuracy: {gb_test_accuracy:.4f}")
print(f"Validation Accuracy: {gb_validation_accuracy:.4f}")
------------------------------------------------------------
Training Accuracy: 0.7532
Test Accuracy: 0.5557
Validation Accuracy: 0.7574
```
![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone5/graphs/gm_fitting_graph.png)

## Discussion

## Conclusion

## Statement of Collaboration