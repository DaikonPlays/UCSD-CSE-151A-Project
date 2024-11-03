# UCSD-CSE-151A-Project

# [Click here for GitHub Repo](https://github.com/DaikonPlays/diet-warriors/tree/main)

## Team Members
| Name | GitHub |
|------|--------|
| Kevin Yan   | [https://github.com/DaikonPlays](https://github.com/DaikonPlays)  |
| Luffy Saito | [https://github.com/rsa1to](https://github.com/rsa1to) |
| Skyler Goh  | [https://github.com/SkylerGoh](https://github.com/SkylerGoh) |
| Phillip Wu  | [https://github.com/philliptwu](https://github.com/philliptwu) |

## Data exploration:

### # of observations

There are 7806 rows in our all_diets file, which means we have a total of 7806 observations on different diets. We have six total features: diet type, recipe name, cuisine, protein, carbs, and fats. Our target feature is the diet type, and the main independent features are protein, carb, fats, and cuisine.

For cuisine types, there are a bunch of categories we account for: ['american' 'south east asian' 'mexican' 'chinese' 'mediterranean'
'italian' 'french' 'indian' 'nordic' 'eastern europe' 'central europe'
'kosher' 'british' 'caribbean' 'south american' 'middle eastern' 'asian'
'japanese' 'world'].
The diet types are our main target feature and there are five classes: ['paleo' 'vegan' 'keto' 'mediterranean' 'dash'].
The rest of the features are numbers and are no discrete.

### Data distribution

There are 1522 obersavtions for vegan, 1274 observations for paleo, 1753 observations for mediterranean, 1512 observations for keto, 1745 observations for dash

## Initial Pre-processing

### Scale

Our two categoriacal classes: diet type and cuisine, just have their own distinct categories for classification.

For protein, fat, and carb, the means are pretty different: 83.23, 117.33, and 152.12 respectively. Furthermore, the stds are also very different: 89.8, 122.1, and 185.91 respectively. This indicates the distribution of data to be very different. Furthermore, the ranges are different too: [0, 1273], [0, 1930.24], and [0.6, 3405.55]. All these indicate that we need to normalize the data. We can do this using either Z-score normalization or Min Max normalization, so features are within a standard deviation of 1 with each other or are within the values 0 and 1.

### Dealing with missing data and null values

There are no missing data and null values. 

### Dropping unneccesary data

There are three features that we are considering dropping, which are cuisine name, extraction date, and time because they don't seem to be that impactful on our overall outcome.

### Classification encoding
For our attributes that are classes: Diet_type and Cuisine_type, we will have to encode them either using ordinal encoding or one-hot encoding for classification models to understand them.

