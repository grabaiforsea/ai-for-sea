# Grab aiforsea
# 
Grab AIFORSEA Safety Challenge

# Instructions :
1. Prepare full training set (rename to "train_full.csv") and test set (rename to "test_full.csv"), please uncomment some lines from "init_full_set.py" if you have done this
2. Run adaboost_model.py (Uncomment some parts for inference/testing purposes)
3. Run final_inference.py (Uncomment some parts for inference/testing purposes)

# Feature Engineering
Since acceleration and gyroscopic readings are read as vector in one direction, we can calculate the norm of the vectors from combinations of vectors.

# Simple write-up :
Instead of generating/engineered features from aggregating readings from the given file we train a model to identify whether a set of observations(from given features and my features) will lead to dangerous driving, without the knowledge of the bookingID.<br>
<br>
Using adaboost with extratrees classifier I was able to get 0.733 of stratified k-fold roc score in relatively short time of training.
<br><br>
Next, for each set of observations we generate probabilities of dangerous driving.
<br><br>
Finally, we simply aggregate the readings from similar bookingID by taking the average and we can get the probability for a trip to be dangerous or not.

# To-Do :
1. Use argparser to improve interface
2. Calculate Feature importances
