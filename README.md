# UCSD-CSE-151A-Project

# [Click here for GitHub Repo](https://github.com/DaikonPlays/diet-warriors/tree/main)

## Team Members
| Name | GitHub |
|------|--------|
| Kevin Yan   | [https://github.com/DaikonPlays](https://github.com/DaikonPlays)  |
| Luffy Saito | [https://github.com/rsa1to](https://github.com/rsa1to) |
| Skyler Goh  | [https://github.com/SkylerGoh](https://github.com/SkylerGoh) |
| Phillip Wu  | [https://github.com/philliptwu](https://github.com/philliptwu) |

## MS2: Data Exploration:
#### [Milestone 2 Notebook](https://github.com/DaikonPlays/diet-warriors/blob/Milestone2/src/data_exploration.ipynb)

### # of Observations

There are 7806 rows in our all_diets file, which means we have a total of 7806 observations on different diets. We have six total features: diet type, recipe name, cuisine, protein, carbs, and fats. Our target feature is the diet type, and the main independent features are protein, carb, fats, and cuisine.

For cuisine types, there are a bunch of categories we account for: ['american' 'south east asian' 'mexican' 'chinese' 'mediterranean'
'italian' 'french' 'indian' 'nordic' 'eastern europe' 'central europe'
'kosher' 'british' 'caribbean' 'south american' 'middle eastern' 'asian'
'japanese' 'world'].
The diet types are our main target feature and there are five classes: ['paleo' 'vegan' 'keto' 'mediterranean' 'dash'].
The rest of the features are numbers and are no discrete.

### Data Distribution

There are 1522 obersavtions for vegan, 1274 observations for paleo, 1753 observations for mediterranean, 1512 observations for keto, 1745 observations for dash

## Initial Pre-Processing Plans

### Scale

Our two categoriacal classes: diet type and cuisine, just have their own distinct categories for classification.

For protein, fat, and carb, the means are pretty different: 83.23, 117.33, and 152.12 respectively. Furthermore, the stds are also very different: 89.8, 122.1, and 185.91 respectively. This indicates the distribution of data to be very different. Furthermore, the ranges are different too: [0, 1273], [0, 1930.24], and [0.6, 3405.55]. All these indicate that we need to normalize the data. We can do this using either Z-score normalization or Min Max normalization, so features are within a standard deviation of 1 with each other or are within the values 0 and 1.

### Dealing with missing data and null values

There are no missing data and null values. 

### Dropping Unneccesary Data

There are three features that we are considering dropping, which are cuisine name, extraction date, and time because they don't seem to be that impactful on our overall outcome.

### Classification Encoding
For our attributes that are classes: Diet_type and Cuisine_type, we will have to encode them either using ordinal encoding or one-hot encoding for classification models to understand them.

## MS3: Data Pre-Processing
#### [Milestone 3 Notebook](https://github.com/DaikonPlays/diet-warriors/blob/Milestone3/src/diet_classifer.ipynb) 

### Conclusion
Our initial logistic regression model yielded surprisingly low accuracy at 40%. Through comparing MSE values between training (3.67) and validation (4.0) sets, we determined that while overfitting is not an issue due to their similar values, the high MSE relative to our label encoding range (0-4) indicates significant underfitting. This poor baseline performance suggests the need to explore alternative approaches. We propose implementing Support Vector Machines (SVM) as our next model, given their strength in handling non-linear relationships. This capability is particularly relevant for our dataset, where diet types and their features show considerable overlap. SVM's ability to create more sophisticated decision boundaries could potentially provide a more accurate classification of our dietary data.

## MS4: Second Model
#### [Milestone 4 Notebook](https://github.com/DaikonPlays/diet-warriors/blob/Milestone4/src/diet_classifer.ipynb) 

### Model evaluations and training and test error
For our new models, we tried two differnet types: XGBoost and SVC. For XGBoost, we used a simple XGBoost model to see if there were any changes in accuracy. After, we tried to do a base SVC model, which did show some improvment, up to mid 50s% for testing and training. After, we added code to fine-tune the SVC model parameters to test different combinations of the regularization parameter C and the kernel coefficient γ to see if there were any further improvements in the result. But, the improvement was insignificant. When looking at the training and test error from our initial model in MS3 and ours now in MS4, we saw an overall 15% improvement in accuracy, thus resulting in less error.

### Where does your model fit in the fitting graph? 

![Fitting Graph](https://github.com/DaikonPlays/diet-warriors/blob/Milestone4/graphs/gb_fitting_graph.png)

Based on our results, it seems that our model is likely underfitting the data, as both the training (59%) and validation (60%) accuracies are relatively low, suggesting the model isn't effectively capturing the underlying patterns. We hypothesize that this underfitting could stem from several factors. First, the default XGBoost settings may not be optimal for our dataset, potentially resulting in insufficient model complexity. Additionally, the features we are using—fat, carbs, protein, and cuisine type—might not provide enough distinguishing power to differentiate between diet types. Second, there could be data imbalance in the dataset, with certain diet types being overrepresented, which could cause the model to struggle with learning patterns for less frequent classes, leading to poor generalization and lower test accuracy.

### What are the next models you are thinking of and why?

After reviewing our current model's performance, we are considering Random Forest and Neural Networks as the next potential models. Random Forest is a robust, versatile algorithm that is less sensitive to overfitting compared to individual decision trees. Given that our XGBoost model performed slightly better on the validation set than on the training set, this may indicate overfitting to validation-specific characteristics. Random Forest, by aggregating multiple weak learners, might help reduce this issue and better capture complex relationships and interactions in the data. Additionally, Neural Networks could be another promising approach, especially for modeling the intricate patterns in diet type classification. Since diet types may share similar features, making them difficult to distinguish, we believe a neural network could learn these complex relationships more effectively. We plan to start with a simple feedforward network as a baseline and further refine it based on the results.

### Conclusion

Our initial logistic regression model achieved relatively low accuracies around 40%, indicating significant underfitting of the data. We then implemented a Support Vector Classifier (SVC) as our second approach, which showed modest improvement with training accuracy at 59%, testing at 54%, and validation at 60%. Despite this improvement, the consistently low performance across all metrics suggests that our model still fails to capture the underlying complexity of the data.
The similar performance between training (59%) and validation (60%) sets reveals that our current model's limitations stem from both architectural constraints and data characteristics. Our feature set—consisting of fat, carbs, protein, and cuisine type—may lack the discriminative power needed for effective diet classification. Additionally, potential class imbalances in our dataset could be hampering the model's ability to learn patterns for underrepresented diet types.
To address these limitations, we propose two promising approaches: Random Forest and Neural Networks. Random Forest's ensemble methodology could better handle complex feature interactions while maintaining robustness against overfitting. Neural Networks offer the potential to learn subtle patterns in diet classifications, particularly useful for distinguishing between similar diet categories. We will also explore feature engineering and address class imbalance issues to enhance model performance. Through these refinements, we aim to develop a more sophisticated and accurate diet classification system that better captures the nuances in our data.


### Predictions of correct and FP and FN from test dataset
Code for this can be found in our MS 4 branch
| Class            | True Positives | False Positives | True Negatives | False Negatives |
|-------------------|----------------|-----------------|----------------|-----------------|
| Dash             | 127            | 131             | 1084           | 220             |
| Keto             | 203            | 186             | 1083           | 90              |
| Mediterranean    | 262            | 114             | 1096           | 90              |
| Paleo            | 57             | 104             | 1189           | 212             |
| Vegan            | 198            | 180             | 1081           | 103             |
