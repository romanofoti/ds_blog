---
layout: post
title: Machine learning from Hurricane Sandy
---

In this post, we will look on how it is possible to use machine learning to identify areas that are vulnerable to natural disasters. The application is specifically tailored to analyze the damage patterns caused by Hurricane Sandy as it hits the East Coast in the Fall 2012.

As the dataset used was protected by an NDA, I will not be able to fully disclose it, so this write-up will discuss the approach only conceptually and draw insights from a few results.

## Objectives
Identify the primary factors of coastal vulnerability and develop a model to assess and predict expected damages from extreme events. More in detail:
Explore the physical and climatic factors determining (or contributing to) NYC buildings damage during Sandy
Use machine learning to model and possibly predict expected damages from future Sandy-like events.

## Datasets (covered by NDA)
NYC’s physical and social community characteristics as well as climatic risks
A categorical characterization of building damages - ranging from 0 (not damaged) to 5 (destroyed and flooded)

## Exploratory analysis
First, let us see what the distribution of damages looks like and perhaps explore some preliminary relationships between predictors and damages, as well as analyzing the correlation among predictors.

## fg

sdrg

<img src="../images/sandi_post/Figure_1.jpg" alt="Drawing" style="width: 200px;"/>

sgdsfgh


![png]({{ site.url }}/images/sandi_post/Figure_1.png)

## Approach and Models
A number of machine learning (ML) classifiers were used to try to capture the relationships between predicting factors and damage[1]:
Logistic Regression
k-Nearest Neighbor (k-NN) classifier
Gradient Boosting Trees
Random Forest
Support Vector Machines (SVM)
Model performance was compared using precision, recall and f-1 scores*:

A number of machine learning (ML) classifiers were used to try to capture the relationships between predicting factors and damage[1]:
Logistic Regression
k-Nearest Neighbor (k-NN) classifier
Gradient Boosting Trees
Random Forest
Support Vector Machines (SVM)
Model performance was compared using precision, recall and f-1 scores*:

ML were cross validated using 80% of the data as training set and 20% as test set. ML algorithms were optimized to address class imbalance and to improve performance. All validation metrics were adapted for multi-class and plotted in Figure 2 for a set of 100 cross-validated samples. 

# INSERT FIGURER

## Model Validation
The Random Forest classifier appeared to be the most accurate and robust and was chosen for further development. Validation based on Receiver Operating Characteristic (ROC) curve and confusion matrix is provided below.

# INSERT FIGURE


The GINI index[1], which is the best splitting criterion used in Random Forest trees, was used to evaluate the relative importance of damage predictors for the trained Random Forest classifier and for each of the 100 cross validated runs, as reported in Figure 4.

# INSERT GINI FIGURE
# INSERT VALIDATION FIGURE

##Predictions

# INSERT PREDICTION FIGURE

## Conclusion
The primary factor determining building damages in NYC during hurricane Sandy is storm surge, although correlation among features prevents us from effectively isolating their relative impact.
The proximity of GI (e.g. parks, trees, wetlands) is identified by the classifiers as a relevant predictor of the damage sustained by buildings.
Rockaways, eastern Staten Island and north-western Staten Island are the most vulnerable areas to hurricane-like events. Current wetlands and areas where historic wetlands were present are among most vulnerable sites.
This approach can be used to predict damages for future events and/or to assess the effect of mitigation measures.
