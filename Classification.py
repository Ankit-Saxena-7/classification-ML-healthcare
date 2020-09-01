# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.ensemble import VotingClassifier

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

# METHOD II: Ensemble models

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

