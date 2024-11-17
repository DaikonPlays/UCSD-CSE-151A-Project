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