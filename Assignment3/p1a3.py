from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix


def main():

    X, y = load_svmlight_file("genres.libsvm")
    X = X.toarray()

    clf1 = GaussianNB()
    clf1.fit(X, y)

    print("Accuracy for GaussianNB: ")
    print(clf1.score(X, y))

    pred1 = clf1.predict(X)
    print("Confusion Matrix for GaussianNB: ")
    print(confusion_matrix(y, pred1))
    print("\n")


    clf2 = SVC()
    clf2.fit(X, y)

    print("Accuracy for SVC: ")
    print(clf2.score(X, y))

    pred2 = clf2.predict(X)
    print("Confusion Matrix for SVC: ")
    print(confusion_matrix(y, pred2))
    print("\n")

    clf3 = DecisionTreeClassifier()
    clf3.fit(X, y)

    print("Accuracy for Decision Tree Classifier: ")
    print(clf3.score(X, y))

    pred3 = clf3.predict(X)
    print("Confusion Matrix for Decision Tree Classifier: ")
    print(confusion_matrix(y, pred3))


if __name__ == "__main__":
    main()