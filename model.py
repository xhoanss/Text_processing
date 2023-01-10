import json
from numpy import mean
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn import linear_model, preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_eng import listings
from pandas import Series
import matplotlib.pyplot as plt

df1 = pd.read_csv("listings.csv")
id = df1.iloc[:,0]

#score features
review_scores_rating = df1["review_scores_rating"]
review_scores_accuracy = df1["review_scores_accuracy"]
review_scores_cleanliness = df1["review_scores_cleanliness"]
review_scores_checkin = df1["review_scores_checkin"]
review_scores_communication = df1["review_scores_communication"]
review_scores_location = df1["review_scores_location"]
review_scores_value = df1["review_scores_value"]

#矩阵的顺序和listing不一样，处理成一样的
def process_matrix(reviews_bag):
    reviews = {}
    for i in id:
        if (str(i) in reviews_bag.keys()):
            reviews.setdefault(i, [])
            reviews[i].append(reviews_bag[str(i)])
        else:
            continue
    re = []
    for i in reviews.values():
        re.append(i[0][0])
    return re

def process_matrix_em(comment_all):
    comments_all = {}
    for i in id:
        if (str(i) in comment_all.keys()):
            comments_all.setdefault(i, [])
            comments_all[i].append(comment_all[str(i)])
        else:
            continue
    docs = []
    for i in comments_all.values():
        docs.append(i)
    return docs

#把listing的feature中不在评论里的房子删除掉
def NotIn_listing(feature,reviews_bag):
    for index,values in id.items():
        if (str(values) in reviews_bag.keys()):
            continue
        else:
            feature = feature.drop(index)
    return feature

#对每个需要预测的分数，取平均值，大于平均值的是1，小于平均值的是0
def process_score(feature):
    score=[]
    for i in feature:
        score.append(i)
    s_mean = mean(score)
    label = []
    for i in score:
        if i >= s_mean:
            label.append(1)
        else:
            label.append(0)
    return label



#############################################################################
#one hot
#for model
# with open("reviews_bag_1000.json", 'r', encoding='UTF-8') as f:
#     reviews_bag = json.load(f)
# one_hot = process_matrix(reviews_bag)
#
# #for review score rating model features
# review_scores_rating.dropna(inplace=True)
# review_scores_accuracy = NotIn_listing(review_scores_accuracy,reviews_bag).fillna(4.5)
# review_scores_cleanliness = NotIn_listing(review_scores_cleanliness,reviews_bag).fillna(4.5)
# review_scores_checkin = NotIn_listing(review_scores_checkin,reviews_bag).fillna(4.5)
# review_scores_communication = NotIn_listing(review_scores_communication,reviews_bag).fillna(4.5)
# review_scores_location = NotIn_listing(review_scores_location,reviews_bag).fillna(4.5)
# review_scores_value = NotIn_listing(review_scores_value,reviews_bag).fillna(4.5)
#
# #把连续的变成分离的，变成classifier问题
# label_encoder = preprocessing.LabelEncoder()
# list_score = label_encoder.fit_transform(review_scores_value)
# #X = np.column_stack((host_is_superhost,number, re))  # let column become an array
#
# x_train, x_test, y_train, y_test = train_test_split(one_hot, list_score, test_size=0.2)
#
# #model = linear_model.LinearRegression()
# model = DummyRegressor()
# #model = linear_model.LogisticRegression(penalty="l2",C=0.001)
# model.fit(x_train,y_train)
# predict = model.predict(x_test)
# print(model.score(x_train, y_train))
# print(model.score(x_test,y_test))
# print(mean_squared_error(list(y_test), predict))
# print(r2_score(list(y_test), predict))
##############################################################################

#######embedding
from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

##把字典的顺序调整并存入
with open("comment_all.json", 'r', encoding='UTF-8') as f:
    comment_all = json.load(f)
docs = process_matrix_em(comment_all)
labels = process_score(NotIn_listing(review_scores_rating,comment_all))


vocab_size = 5000
encoded_docs = [one_hot(d[0], vocab_size) for d in docs]
max_length = 2000
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
input = Input(shape=(2000,))
x = Embedding(vocab_size, 8, input_length=max_length)(input)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
xtrain, xtest, ytrain, ytest = train_test_split(padded_docs, np.array(labels), test_size=0.2)
history = model.fit(xtrain, ytrain, epochs=8, batch_size=32)

# evaluate the model
loss, accuracy = model.evaluate(xtest, ytest, verbose=0)
loss_test, accuracy_test = model.evaluate(xtest, ytest, verbose=0)
print('Accuracy: %f' % (accuracy * 100))


loss_values = history.history['loss'][1:]
epochs = range(0, len(loss_values) + 1)
plt.plot(epochs, history.history['loss'], 'bo', label='Loss')
plt.plot(epochs, history.history['acc'], 'b', label='Accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy and Loss values')
plt.legend()

plt.show()
