from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
algl = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic['Survived'].iloc[train]
    algl.fit(train_predictors, train_target)
    test_predictions = algl.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)


predictions = np.concatenate(predictions, axis=0)
predictions[predictions>0.5] = 1
predictions[predictions<0.5] = 0
accuracy = sum(predictions[predictions==titanic['Survived']]) / len(predictions)
print(accuracy)

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg2 = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg2, titanic[predictors], titanic['Survived'], cv=3)
print(scores.mean())

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
print(scores.mean())


from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic['Survived'])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
predictors = ['Pclass', 'Sex', 'Fare', 'Title']
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)

print(scores.mean())


from sklearn.ensemble import GradientBoostingClassifier

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']],
    [LogisticRegression(random_state=1), ['Pclass', 'Sex', 'Fare', 'Familysize', 'Age', 'Embarked']]
]
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions<=0.5] = 0
    test_predictions[test_predictions>0.5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions==titanic['Survived']]) / len(predictions)
print(accuracy)
