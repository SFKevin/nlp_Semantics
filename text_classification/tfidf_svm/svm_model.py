from sklearn.svm import SVC
import sklearn.metrics as meth
from tfidf_svm import data_utils
def train(x_train,y_train):
    model=SVC()
    model.fit(x_train,y_train)
    return model

def predict(model,x_dev,y_dev):
    y_pre=model.predict(x_dev)
    acc=meth.accuracy_score(y_dev,y_pre)
    f1score=meth.f1_score(y_dev,y_pre)
    recall=meth.recall_score(y_dev,y_pre)
    return acc,f1score,recall

filepath_pos="E:\\data\\rt-polaritydata\\rt-polarity.pos"
filepath_neg="E:\\data\\rt-polaritydata\\rt-polarity.neg"
if __name__ == '__main__':
    x_train,y_train,x_dev,y_dev=data_utils.load_data_and_labels(filepath_pos,filepath_neg)
    model=train(x_train,y_train)
    acc,f1score,recall=predict(model,x_dev,y_dev)
    print("accuracy: {0:.3f}, f1score: {0:.3f}, recall: {0:.3f}".format(acc,f1score,recall))
