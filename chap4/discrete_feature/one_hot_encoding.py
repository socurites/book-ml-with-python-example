import mglearn
import os
import pandas as pd
data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, 'adult.data'),
    header=None,
    index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'martial-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
print(data.head())

print(data.gender.value_counts())
print(data.workclass.value_counts())


"""One-Hot Encoding
"""
print("Original Feature:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Dummies Feature:\n", list(data_dummies.columns), "\n")


# Split X from y
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']

X = features.values
y = data_dummies['income_ >50K'].values

print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

# Classifiy with LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logit = LogisticRegression().fit(X_train, y_train)

print("Test Accuracy: {:.2f}".format(logit.score(X_test, y_test)))


"""pd.get_dummies
- pandas의 get_dummies 함수는 숫자 특성은 모두 연속형이라고 생각해서 가변수를 만들지 않는다.
- 1) scikit-learn의 OneHotEncoder를 사용
- 2) DataFrame에 있는 숫자값을 범주형인 경우 문자열로 변환
"""