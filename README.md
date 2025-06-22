# Logistic Regression

`(default) solver=lbfgs, penalty=l2  `
```
[2900]
              precision    recall  f1-score   support  
           0       0.99      0.88      0.94      1324  
           1       0.18      0.82      0.29        40  

Confusion Matrix:[[1171  153]  
                 [   7   33]]  
ROC AUC Score: 0.93  
PR AUC score: 0.30  
Accuracy_score: 0.88  
best_f1: 0.42  
best_thresh: 0.81  
```
`solver="liblinear",penalty="l2"  `
```
[50]  
              precision    recall  f1-score   support  
           0       0.99      0.88      0.94      1324  
           1       0.17      0.80      0.28        40  

Confusion Matrix:  [[1170  154]  
                   [   8   32]]  
ROC AUC Score: 0.93  
PR AUC score: 0.30  
Accuracy_score: 0.88  
best_f1: 0.41  
best_thresh: 0.80  
```
`solver="saga",penalty="l1"  `
```
[2770]  
              precision    recall  f1-score   support  
           0       0.99      0.88      0.94      1324  
           1       0.17      0.80      0.28        40  

Confusion Matrix:  [[1171  153]  
                   [   8   32]]  
ROC AUC Score: 0.93  
PR AUC score: 0.30  
Accuracy_score: 0.88  
best_f1: 0.42  
best_thresh: 0.81  
```
`solver="liblinear",penalty="l1"  `
```
[35]  
              precision    recall  f1-score   support  
           0       0.99      0.89      0.94      1324  
           1       0.18      0.78      0.29        40  

Confusion Matrix:  [[1179  145]  
                   [   9   31]]  
ROC AUC Score: 0.93  
PR AUC score: 0.31  
Accuracy_score: 0.89  
best_f1: 0.38  
best_thresh: 0.86  
```
`best_f1 for logistic regression is 0.42`  
`best model= (default) solver=lbfgs, penalty=l2  `

# Decision Tree 

`(default) max_depth=None`  
```
           precision    recall  f1-score   support  
           0       0.98      0.95      0.97      1324  
           1       0.23      0.45      0.31        40  

Confusion Matrix:  [[1264   60]  
                   [  22   18]]  
ROC AUC Score: 0.70  
PR AUC score: 0.12  
Accuracy_score: 0.94  
best_f1: 0.31  
best_thresh: 0.01  
```
```
max_depth=3, best_f1: 0.35  
             best_thresh: 0.82  
max_depth=5, best_f1: 0.36  
             best_thresh: 0.94  
max_depth=6, best_f1: 0.38  
             best_thresh: 0.94            
max_depth=7, best_f1: 0.36  
             best_thresh: 0.96  
max_depth=10,best_f1: 0.33  
             best_thresh: 0.51  
```
`best_f1 for decision tree is 0.38`  
`best model= max_depth=6  `
# RandomForestClassifier

## *(max_depth=6)*  
```
n_estimators=100, best_f1: 0.48  
                  best_thresh: 0.84
```
```
n_estimators=150, best_f1: 0.49  
                  best_thresh: 0.84  

precision    recall  f1-score   support  
           0       0.99      0.92      0.95      1324  
           1       0.22      0.78      0.34        40  

Confusion Matrix: [[1213  111]  
                  [   9   31]]  
ROC AUC Score: 0.94  
PR AUC score: 0.42  
Accuracy_score: 0.91  
best_f1: 0.49  
best_thresh: 0.85           
```
```
n_estimators=200, best_f1: 0.48  
                  best_thresh: 0.86  
```
## *(max_depth=8, n_estimators=150)*  
```
         precision    recall  f1-score   support  
           0       0.99      0.93      0.96      1324  
           1       0.25      0.75      0.38        40  

Confusion Matrix:  
  [[1235   89]  
 [  10   30]]  
ROC AUC Score: 0.95  
PR AUC score: 0.46  
Accuracy_score: 0.93  
best_f1: 0.50  
best_thresh: 0.82   
```
`best_f1 for RandomForestClassifier is 0.50`  
`best model= (max_depth=8, n_estimators=150)  `
## *GridSearchCV* 
```
params={  
    "max_depth":[6,8,10],    
    "max_features":["sqrt","log2"],  
    "n_estimators":[100,150,200],  
    "bootstrap":[True],  
}  

{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 200}   
best_model.best_score_= 0.9722797806093901  
              precision    recall  f1-score   support  
           0       0.99      0.95      0.97      1324  
           1       0.29      0.72      0.42        40  

Confusion Matrix:  [[1254   70]  
                   [  11   29]]  
ROC AUC Score: 0.94  
PR AUC score: 0.45  
Accuracy_score: 0.94  
best_f1: 0.48  
best_thresh: 0.81  
```

# XGBoostClassifier
```
n_estimators=500,
learning_rate=0.05,
max_depth=5,
scale_pos_weight=scale_pos_weight,
eval_metric="aucpr",
early_stopping_rounds=30

            precision    recall  f1-score   support
           0       0.99      0.94      0.97      1324
           1       0.27      0.70      0.39        40

Confusion Matrix: [[1249   75]
                  [  12   28]]
ROC AUC Score: 0.91
PR AUC score: 0.39
Accuracy_score: 0.94
best_f1: 0.45
best_thresh: 0.81
```
