#5 classification model from scikit learn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

#K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

import numpy as np
from numpy import genfromtxt
#import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV

#Deep learning Librarys
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

'''
def nn_classifier_train(train_data,label_data,hidden_layer_sizes):
    train_holder = tf.placeholder(tf.float32, [None, train_data.shape[0]])  # 28x28
    label_holder = tf.placeholder(tf.float32, [None, 1])
    fcn = []
    input =
    for i in range(len(hidden_layer_sizes)):
        if(i == 0):
            fcn[i] = tf.contrib.fully_connected(inputs=input, units=hidden_layer_sizes[i], activation=tf.nn.relu)
        fcn[i] = tf.contrib.fully_connected(inputs=fcn[i-1], units=hidden_layer_sizes[i], activation=tf.nn.relu)
    output = tf.contrib.fully_connected(inputs=fcn[len(hidden_layer_sizes)], units=1)
'''

def create_model():
    # create model
    model = Sequential()
    #model.add(Dense(64, input_dim=50, activation='relu'))
    model.add(Dense(64, input_dim=57, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_data = np.genfromtxt('traindata.csv', delimiter=',')
    print(train_data.shape)
    label_data = np.genfromtxt('trainlabel.csv', delimiter=',')
    test_data = np.genfromtxt('testdata.csv', delimiter=',')

    ## 1 . Feature selection and dimension reduction

    # train_data = VarianceThreshold(threshold=0.01).fit_transform(train_data)
    # k = np.shape(train_data)[1]
    # print(k)
    # # # train_data = SelectKBest(chi2, k-4).fit_transform(train_data, label_data)
    # # # k = np.shape(train_data)[1]
    # # # print(k)
    # sfm = SelectFromModel(LogisticRegression(penalty='l1',C=1))
    # sfm.fit(train_data, label_data)
    # train_data = sfm.transform(train_data)

    #pca = PCA(n_components=55)
    #train_data = pca.fit_transform(train_data)
    #lda = LinearDiscriminantAnalysis(n_components=40)
    #train_data = lda.fit_transform(train_data, label_data)
    scores = [[] for i in range(5)]

    # #test model 1 : K nearest neighbors
    # ng_num = np.arange(1,100)
    # KNNscore = np.zeros(99)
    # for i in ng_num:
    #     KNNmodel = KNeighborsClassifier(n_neighbors=i)
    #     KNNscore[i-1] = np.mean(cross_val_score(KNNmodel,train_data,label_data,cv=5))
    # print(ng_num)
    # print(KNNscore)
    # plt.plot(ng_num,KNNscore,'bx-')
    # plt.xlabel('K number')
    # plt.ylabel('validation score')
    # plt.title('KNNmodel selection')
    # plt.show()
    #
    # KNNmodel = KNeighborsClassifier()
    # scores[0] = cross_val_score(KNNmodel,train_data,label_data,cv=5)
    #
    #test model 2 : Logistic regression
    c = [0.0001,0.001,0.01,0.1,1,10,100]
    LRscore = np.zeros(len(c))
    j = 0
    for i in c:
        logisticModel = LogisticRegression(penalty='l1',C=i)
        LRscore[j] = np.mean(cross_val_score(logisticModel,train_data,label_data,cv=5))
        j = j+1
    print(c)
    print(LRscore)
    # plt.plot(c,LRscore,'bx-')
    # plt.xlabel('penalty')
    # plt.ylabel('validation score')
    # plt.title('LR Model selection')
    # plt.show()
    # #logisticModel = LogisticRegression(penalty='l2')
    # #scores[1] = cross_val_score(logisticModel,train_data,label_data,cv=5)
    #
    #test model 3 : Neutral network
    #NNModel = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5000,100), random_state=1,max_iter=500)
    tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    NNModel = KerasClassifier(build_fn=create_model,epochs=1200, batch_size=150,verbose=0)
    cv = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    #NNscore = cross_val_score(NNModel,train_data,label_data,fit_params={'callbacks': [tbCallback]},cv=cv)
    NNModel.fit(train_data,label_data)
    prediction = NNModel.predict(test_data)
    prediction = np.array(prediction)
    print(prediction)
    np.savetxt("prediction.csv", prediction, delimiter=",")
    #print('MLPClassifier validation score : ',NNscore)


    #test model 4 : SVM
    # c = [1]
    # SVMscore = np.zeros(len(c))
    # j = 0
    # for i in c:
    #     svmModel = SVC(C=i,kernel='linear')
    #     SVMscore[j] = np.mean(cross_val_score(svmModel,train_data,label_data,cv=5))
    #     j = j+1
    # print(c)
    # print(SVMscore)
    # plt.plot(c,SVMscore,'bx-')
    # plt.xlabel('penalty')
    # plt.ylabel('validation score')
    # plt.title('SVM Model selection')
    # plt.show()
0

    # scores = [x[1] for x in clf.grid_scores_]
    # scores = np.array(scores).reshape(len(cS), len(gammaS))
    #
    # for ind, i in enumerate(Cs):
    #     plt.plot(Gammas, scores[ind], label='C: ' + str(i))
    # plt.legend()
    # plt.xlabel('Gamma')
    # plt.ylabel('Mean score')
    # plt.show()
    # svmModel = SVC(kernel='rbf', probability=True)
    # scores[3] = cross_val_score(svmModel,train_data,label_data,cv=5)
    # print('SVMClassifier validation score : (5 fold)',np.mean(scores[3]))


    # #test model 5 :Adaboost
    # c = np.arange(50,1000,100)
    # adascore = np.zeros(len(c))
    # j = 0
    # for i in c:
    #     adaModel = AdaBoostClassifier(n_estimators=i)
    #     adascore[j] = np.mean(cross_val_score(adaModel,train_data,label_data,cv=5))
    #     j = j+1
    # print(c)
    # print(adascore)
    # plt.plot(c,adascore,'bx-')
    # plt.xlabel('n_estimators')
    # plt.ylabel('validation score')
    # plt.title('ada Model selection')
    # plt.show()
    # #adaModel = AdaBoostClassifier()
    # #treeModel.fit(training_x,training_y)
    # #scores[4] = cross_val_score(adaModel,train_data,label_data,cv=5)
    #

'''
    #print(scores)
    for score in scores:
        print(np.mean(score))
    K = range(1,10)
'''
