---
layout: post
title: Machine learning from Hurricane Sandy
tags: machine-learning 
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

{% include image.html img="images/sandy_post/Figure_1.png" title="Damage - Canopy" width="600" %}

{% include image.html img="images/sandy_post/Figure_2.png" title="Damage - Surge" width="600" %}

{% include image.html img="images/sandy_post/Figure_3.png" title="Damage - Elevation" width="600" %}

{% include image.html img="images/sandy_post/Figure_4.png" title="Damage - Distance from Shore" width="600" %}

We notice that damages - here expressed on a scale 0 to 5 - are generally greater for: 1) short distance from the shore; 2) low elevation; 3) high surge with respect to the ground level; 4) when the density of street canopy is smaller. 

## Approach and Models
The above preliminary analyses suggest that there is a set of meaningful predictors whose relationship with building damage can be utilized to build models and eventually make predictions.
Here, I explore a number of machine learning classifiers to make a preliminary assessment of the potential of a machine learning approach:
 - Logistic Regression
 - k-Nearest Neighbor (k-NN) classifier
 - Gradient Boosting Trees
 - Random Forest
 - Support Vector Machines (SVM)

I then use precision, recall and f-1 scores to compare model performances.

Here, I use an 80-20 train-test split to cross validate (using 100 random runs each) the above models. Resampling was used to address class imbalance and hyperparameters were tuned to improve performance. All validation metrics were adapted for multi-class classification. 

{% include image.html img="images/sandy_post/Models_comparison.png" title="Model Comparison" width="700" %}

## Model Validation
The Random Forest classifier appeared to be the most accurate and robust and was chosen for further development. Validation based on Receiver Operating Characteristic (ROC) curve and confusion matrix is provided below.

{% include image.html img="images/sandy_post/RandForest_autoweight_Multiclass_ROC.png" title="RF ROC" width="700" %}

{% include image.html img="images/sandy_post/RandForest_CM.png" title="RF CM" width="500" %}

The GINI index, which is the best splitting criterion used in Random Forest trees, can also provide a good measure of the relative importance of the damage predictors for the trained model. Here is the feature importance, ranked according to the GINI index, for the selected model. The bar on the histogram represent one standard deviation as calculated from the sample of 100 cross validated runs.

{% include image.html img="images/sandy_post/RF_GINI_importance.png" title="RF GINI" width="500" %}

Now that we looked at the metrics, let us see how well we predicted the actual damages on the test set for a subset of the study domain. 

{% include image.html img="images/sandy_post/validation.png" title="observations vs predictions" width="800" %}

Well, not every point is perfectly predicted, but the overall picture is spot on!

## Predictions

Last step: how does the rest of the study domain looks like? Here is a map of the Staten Island and Rockaways area.

{% include image.html img="images/sandy_post/predicted_domain.png" title="Predicted domain" width="800" %}

Overall, most of the area is colored in blue, which means that no damage to buildings is expected. However, pockets of vulnerability are clearly present, especially in the Rockaways, eastern Staten Island and north-western Staten Island. This is perfectly in line with the areas most hevily hit by Sandy, suggesting that the approach taken can indeed be used to predict damages for future events and/or to assess the effect of mitigation measures.

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>