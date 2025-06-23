no missing values found  
converted dtype from float64 to float32 and int64 to int8  
then splitting dataframe into features and label  
then using train_test_split and saving all data to joblib

now big problem is that data is highly imbalanced, about only 3% cases are bankrupt, we will not prefer accuracy much here

firstly i thought that all columns of data are scaled and also didn't thought about outliers and got following results:


# LOGISTIC REGRESSION (precison , recall , f1-score)
in log_reg (default) solver-lbfgs, penalty-l2,max_iter-100  
-not using class_weight=balanced (0.08 , 0.02 , 0.03)  
-using class_weight=balanced from now (0.05 , 0.33 , 0.08)  
-now using solver liblinear instead of default lbfgs with l2 penalty(0.05 , 0.35 , 0.08)  
-using l1 penalty for liblinear (0.19 , 0.79 , 0.31)  
-using l1 penalty for solver-saga although very fast (0.04 , 0.44 , 0.07)  
i saved my best model with score 0.31 and 75 selected features filtered by l1 penalty  

now i use predict_proba method on test set
precision,recall,thresholds=precision_recall_curve(y_test,y_proba)  
-max f1_score 0.41 for threshold 0.67 (0.71 , 0.29 , 0.41)  
drawing precision vs thresholds and recall vs thresholds and f1_scores vs thresholds on same figure

## ROC Curve (Receiver Operating Characteristics)
fpr,tpr,threshold=roc_curve(y_test,y_proba)  
-for each value of threshold find f1_score and append in a list  
now drawing fpr vs threshold, tpr vs threshold , f1_scores vs threshold  
fpr-false positive rate(how many normal companies were wrongly predicted as bankrupt), tpr-true positive rate/recall  
plotting fpr vs tpr(graph is fastly increasing which is a good sign)- more tpr for less fpr
roc_auc_score=0.92  
<!-- pending- stratified k fold and cross validation -->

<!-- ## instead of l1, feature selection using l2 penalty -->


# DECISION TREE (max_depth=5)
making function for outputting for custom threshold=0.67  
making function checking f1_score for each value of threshold and find value of optimal threshold in range[0.3,0.8]  
optimal threshold for full range [0,1] is 0.93 while for [0.3,0.8] is 0.76
(0.19 , 0.60 , 0.29) and (0.20 , 0.56 , 0.30) respectively for 0.76 and 0.93
here f1_score is almost equal but recall is higher for 0.76 since missing bankrupt one is more costlier than flagging right one


# RANDOMFORESTCLASSIFIER 
training model with class_weight=balanced with 100 trees, max_depth=None and using threshold testing for probabilities  
using predict method (0.67 , 0.08, 0.15)  
best_thresh=0.3, (0.53 , 0.38 , 0.44)  

using max_depth=5 best_thresh=0.62, (0.33 , 0.56 , 0.42)  
using max_depth=7 best_thresh=0.57, (0.36 , 0.54 , 0.43)  
using max_depth=10 best_thresh=0.47 (0.38 , 0.52 , 0.44)  
using max_depth=8 best_thesh=0.50 (0.37 , 0.56 , 0.45) --it is also same as predict method

# KAGGLE
param_dist={  
    'n_estimators': [100,200,300],  
    'max_depth': [5,10,15,20],  
    'min_samples_split': [10,20,40,60],  
    'min_samples_leaf': [5,10,15,20,25],  
    'max_features': ['sqrt','log2'],  
    'bootstrap': [True]  
}  
model=RandomizedSearchCV(  
    estimator=rnd_clf,  
    param_distributions=param_dist,  
    n_iter=30,  
    scoring="f1",  
    cv=5,  
    n_jobs=-1,  
    verbose=2,  
    random_state=42  
)  
(RandomForestClassifier(class_weight='balanced',   max_depth=20,  
                        min_samples_leaf=10,   min_samples_split=20,  
                        random_state=42),  
 {'n_estimators': 100,  
  'min_samples_split': 20,  
  'min_samples_leaf': 10,  
  'max_features': 'sqrt',  
  'max_depth': 20,  
  'bootstrap': True},  
 0.48194325187761555)  

using gridsearchCV  
    (RandomForestClassifier(class_weight='balanced',   max_depth=15,  
                            min_samples_leaf=10,   min_samples_split=20),  
    {'bootstrap': True,  
    'max_depth': 15,  
    'max_features': 'sqrt',  
    'min_samples_leaf': 10,  
    'min_samples_split': 20,  
    'n_estimators': 100},  
    0.4938175846304932)  

new_param_dist2={  
    'n_estimators': [100,150],  
    'max_depth': [15],  
    'min_samples_split': [15,20,25],  
    'min_samples_leaf': [8,10,12],  
    'max_features': ['sqrt'],  
    'bootstrap': [True]  
}  
best score:  0.4968845265667229  
best params:  {'bootstrap': True, 'max_depth': 15,'max_features': 'sqrt', 'min_samples_leaf': 10, 'min_samples_split': 15, 'n_estimators': 150}  
best model:  RandomForestClassifier  (class_weight='balanced', max_depth=15,  
                       min_samples_leaf=10,   min_samples_split=15,  
                       n_estimators=150)  

# feature selection
for cumulative sum 0.75, 0.90 and 0.95 respectively values are 29, 57 and 72 now i should try for less cumulative sum because this numbers of features are also very high i think     
26 top features are having importances greater than 0.01 i think i should try with 30 features from now  
i trained best model which got f1_score 0.49 with all features and tested on 20% set, f1_score-0.41 while with 30 features i got 0.38
on increasing from 30 to 40 f1-score is now 0.39 now i think i should go to more advanced models           
# notes
liblinear- for smaller datasets, Good for binary classification and supports L1 and L2 regularization  
saga- larger datasets or sparse data, Works with L1, L2, and elasticnet, supports multiclass  
lbfgs- good default, optimized for l2, better for larger data than liblinear

l1 penalty is slow in log_reg bcz forces many coefficients to exactly 0(sparse model), so solver has to perform extra optimization steps to determine which features to drop, and it also introduces non-differentiable steps in cost function

while training with l1 penalty for both saga and liblinear solver i noticed although saga performed very quickly but poor results thus Faster training is not equal to better performance  
SAGA uses stochastic updates which can be less stable for smaller datasets and with class imbalance  
gradient boosting has no native support for class_weight=balanced  
XGBoost- scale_pos_weight=30.9 (num_negative/num_positive)  

error_raise=warn - to skip bad combos bcz if one
parameter combination causes failure(memory error), whole thing may stop

/kaggle/warning- writable path  
/kaggle/input- read only  

❌ Avoid PCA unless you're trying to compress for models like SVM or KNN that suffer in high dimensions, Creates unnamed components, hard to explain  
# ideas 
stratified k fold  
using votingclassifier in the end  

At first glance i thought that all columns of data are scaled and also didn't thought about outliers and used class_weight:"balanced" (No SMOTE) and trained various models  
I will share their results in very last for just comparison how things get changed