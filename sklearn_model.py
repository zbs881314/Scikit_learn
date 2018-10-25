from sklearn import datasets
from sklearn.linear_model import LinearRegression


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target


model = LinearRegression()
model.fit(data_X, data_y)

print(model.coef_) # y=ax + b print out a
print(model.intercept_)  #print out b
print(model.get_params())
print(model.score(data_X, data_y))  #R^2 coefficient of determination
