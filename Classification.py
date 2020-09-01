# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error as MSE, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

SEED = 1

# Import and process data
vDataFrameRaw = pd.read_csv("Data/Indian Liver Patient Dataset (ILPD).csv")

print(vDataFrameRaw.shape)

# Categorical encoding
vCleanupDict = {
    "gender": {"Female": 1, "Male": 2}
}
vDataFrameRaw.replace(vCleanupDict, inplace=True)

# Missing data
print(vDataFrameRaw['alkphos'].isna().sum())
print(vDataFrameRaw['alkphos'].value_counts())
vDataFrameRaw['alkphos'].fillna(vDataFrameRaw['alkphos'].mean(), inplace=True) # https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
print(vDataFrameRaw['alkphos'].isna().sum())

vX = vDataFrameRaw.drop(columns='is_patient')
vY = vDataFrameRaw['is_patient']

# Split data
vXTrain, vXTest, vYTrain, vYTest = train_test_split(vX, vY, test_size=0.2, random_state=SEED, stratify=vY)

# METHOD I: Decision Tree CV 10-Fold
vDecisionTree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.13, random_state=SEED)

# Mean square errors for 10-fold CV
MSEScoresCV = cross_val_score(  vDecisionTree,
                                vXTrain,
                                vYTrain,
                                cv=10,
                                scoring='accuracy', # https://scikit-learn.org/stable/modules/model_evaluation.html
                                n_jobs=-1) # To utilize all CPU processors

print('CV RMSE: {:.2f}'.format((MSEScoresCV.mean())**(1/2)))

# Mean square error by classification
vDecisionTree.fit(vXTrain, vYTrain)
YPredTrain = vDecisionTree.predict(vXTrain)
print('Train RMSE: {:.2f}'.format((MSE(vYTrain, YPredTrain))**(1/2)))

# High bias or high variance?
BaselineRMSE = 0.51 # Assumption
print('Baseline MSE: {:.2f}'.format(BaselineRMSE))

print('Model has high bias')

# METHOD II: Voting classifier

vLogisticRegression = LogisticRegression(random_state=SEED, max_iter=3000)
vKNeighbors = KNeighborsClassifier(n_neighbors=27)
vDecisionTree = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

vClassifierList = [('Logistic Regression', vLogisticRegression), ('K Neighbors', vKNeighbors), ('Decision Tree', vDecisionTree)]

# Iterate over the pre-defined list of classifiers
for vClassifierName, vClassifier in vClassifierList:
    # Fit clf to the training set
    vClassifier.fit(vXTrain, vYTrain)

    # Predict y_pred
    vYPred = vClassifier.predict(vXTest)

    # Calculate accuracy
    vAccuracy = accuracy_score(vYPred, vYTest)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(vClassifierName, vAccuracy))

vVotingClassifier = VotingClassifier(estimators=vClassifierList)
vVotingClassifier.fit(vXTrain, vYTrain)
vYPred = vVotingClassifier.predict(vXTest)
vAccuracy = accuracy_score(vYPred, vYTest)
print('{:s} : {:.3f}'.format('Voting Classifier', vAccuracy))

# METHOD III: Bagging classifier (sampling with repetition)
vBaggingClassifier = BaggingClassifier(base_estimator=vDecisionTree, n_estimators=50, random_state=SEED, oob_score=True)
vBaggingClassifier.fit(vXTrain, vYTrain)
vYPred = vBaggingClassifier.predict(vXTest)
vAccuracy = accuracy_score(vYPred, vYTest)
vOOBAccuracy = vBaggingClassifier.oob_score_
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(vAccuracy, vOOBAccuracy))

# METHOD IV: Random forest
vRandomForestClassifier = RandomForestClassifier(n_estimators=25, random_state=SEED)
vRandomForestClassifier.fit(vXTrain, vYTrain)
vYPred = vRandomForestClassifier.predict(vXTest)
vRMSE = MSE(vYTest, vYPred)**(1/2)
print('Test set RMSE of Random Forest: {:.2f}'.format(vRMSE))

# Create a pd.Series of features importances
vImportances = pd.Series(data=vRandomForestClassifier.feature_importances_,
                        index= vXTrain.columns)

# Sort importances
vImportancesSorted = vImportances.sort_values()

# Draw a horizontal barplot of importances_sorted
vImportancesSorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# METHOD V: Boosting
# Ada Boosting
vAdaBoostClassifier = AdaBoostClassifier(base_estimator=vDecisionTree, n_estimators=180, random_state=SEED)

# Fit ada to the training set
vAdaBoostClassifier.fit(vXTrain, vYTrain)

# Compute the probabilities of obtaining the positive class
vYPredProba = vAdaBoostClassifier.predict_proba(vXTest)[:, 1]
vAdaROCAUC = roc_auc_score(vYTest, vYPredProba)
print('ROC AUC score: {:.2f}'.format(vAdaROCAUC))

# Gradient Boosting
vGradientBoostingClassifier = GradientBoostingClassifier(max_depth=4, n_estimators=180, random_state=SEED)
vGradientBoostingClassifier.fit(vXTrain, vYTrain)
vYPred = vGradientBoostingClassifier.predict(vXTest)
vRMSE = MSE(vYTest, vYPred)**(1/2)
print('Test set RMSE of Gradient Boosting Classifier: {:.2f}'.format(vRMSE))

# Stochastic Gradient Boosting
vStochasticGradientBoostingClassifier = GradientBoostingClassifier(max_depth=4, n_estimators=180, random_state=SEED, subsample=0.9, max_features=0.75)
vStochasticGradientBoostingClassifier.fit(vXTrain, vYTrain)
vYPred = vStochasticGradientBoostingClassifier.predict(vXTest)
vRMSE = MSE(vYTest, vYPred)**(1/2)
print('Test set RMSE of Stochastic Gradient Boosting Classifier: {:.2f}'.format(vRMSE))

# METHOD VI: Hyperparameter tuning of decision tree
print(vDecisionTree.get_params())

vParamsDecisionTree = {
    'max_depth': [2, 3, 4],
    'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]
}

VGridDecisionTree = GridSearchCV(   estimator=vDecisionTree,
                                    param_grid=vParamsDecisionTree,
                                    scoring='roc_auc',
                                    cv=5,
                                    n_jobs=-1)

VGridDecisionTree.fit(vXTrain, vYTrain)
vBestModel = VGridDecisionTree.best_estimator_
vYPredProba = vBestModel.predict_proba(vXTest)[:, 1]
vAdaROCAUC = roc_auc_score(vYTest, vYPredProba)
print('ROC AUC score: {:.2f}'.format(vAdaROCAUC))

# METHOD VII: Hyperparameter tuning of random forest
print(vRandomForestClassifier.get_params())

vParamsRandomForest = {
    'n_estimators': [100, 350, 500],
    'max_features': ['log2', 'auto', 'sqrt'],
    'min_samples_leaf': [2,10,30]
}

VGridRandomForest = GridSearchCV(estimator=vRandomForestClassifier,
                       param_grid=vParamsRandomForest,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

VGridRandomForest.fit(vXTrain, vYTrain)
vBestModel = VGridRandomForest.best_estimator_
vYPred = vBestModel.predict(vXTest)
vRMSE = MSE(vYTest, vYPred)**(1/2)
print('Test set RMSE of Grid CV Random Forest: {:.2f}'.format(vRMSE))