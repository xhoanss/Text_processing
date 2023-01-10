from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

c_range = [0.0001,0.001,0.1,1,10]
    mean = []
    str = []
    for c in c_range:
        model = linear_model.LogisticRegression(penalty='l2', C=c)
        scores = cross_val_score(model,re,list_score,cv=5)
        mean.append(np.array(scores).mean())
        str.append(np.array(scores).std())

    plt.errorbar(c_range,mean,yerr=str)
    plt.xlabel('C range')
    plt.ylabel('F1 score')
    plt.show()

kf = KFold(n_splits=5)
mean_error = []
str_error = []
k_range = [7, 8, 9, 10, 13, 14, 15, 17, 20, 23, 24, 25, 26, 27, 28, 30]
for k in k_range:
    mse_collect = []
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')

    for train, test in kf.split(X):
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        mse_collect.append(mse)

    mean_error.append(np.array(mse_collect).mean())
    str_error.append(np.array(mse_collect).std())

plt.errorbar(k_range, mean_error, yerr=str_error, color='red')
plt.xlabel('k range')
plt.ylabel('Mean Square Error')
plt.title('KNN')
plt.show()