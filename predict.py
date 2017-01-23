import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scrape import MayoDataProvider
from scrape import CauseType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import seaborn as sns
# sns.set_style("whitegrid")

RELEASE = False


class TestGenerator:

    def __init__(self, X, y, type="Exact"):
        genTypes = {"Exact": self._exact_symptoms_generator,
                    "One missing": self._one_missing_generator,
                    "Triangular": self._triangular_generator,
                    "One addition": self._one_additional_generator}
        self.generator = genTypes[type]
        self.X = X
        self.y = y

    def create_data(self, n=10):
        return self.generator(n)

    def _exact_symptoms_generator(self, n):

        ind = np.random.choice(len(self.y), n)
        return (self.X[ind], self.y[ind])

    @staticmethod
    def random_modification(x, n=1):
        px = np.int64(x == (np.sign(n)+1)/2)
        p = px / sum(px)
        ind = np.random.choice(len(x), size=np.abs(n), p=p)
        x[ind] = np.int64(n<0)
        return x

    def _one_missing_generator(self, n):
        ind = np.random.choice(len(self.y), n)
        X = self.X[ind]
        X = np.apply_along_axis(TestGenerator.random_modification, 1, X)
        return X, self.y[ind]

    def _one_additional_generator(self, n):
        ind = np.random.choice(len(self.y), n)
        X = self.X[ind]
        X = np.apply_along_axis(TestGenerator.random_modification, 1, X, -1)
        return X, self.y[ind]

    def _triangular_generator(self, n):
        ind = np.random.choice(len(self.y), n)
        X = self.X[ind]

        maxrem = np.sum(X, axis=1)
        maxadd = maxrem
        maxadd[maxrem == 0] = 1

        mod = np.int64(np.round(np.random.triangular(-maxrem,0,maxadd,size=len(ind))))
        # X = np.apply_along_axis(TestGenerator.remove_one, )
        for m, i in zip(mod, np.arange(len(ind))):
            X[i] = TestGenerator.random_modification(X[i], m)
        return X, self.y[ind]

def test_models(trainX, trainY, testgenerator, models):
    scores = []
    for model in models:
        model.fit(trainX, trainY)

def m_star_distance(x, y):
    return np.sum(np.int64(x > y)) + np.sum(np.int64(x < y))/float(len(x)+1)

def m_star_inverse(x, y):
    return np.sum(np.int64(x > y))/ float(len(x) + 1) + np.sum(np.int64(x < y))

testModels = {
    "kNN manhattan": KNeighborsClassifier(metric="manhattan", n_neighbors=1),
    "kNN M*": KNeighborsClassifier(metric=m_star_distance, algorithm="brute", n_neighbors=1),
    "kNN M*'": KNeighborsClassifier(metric=m_star_inverse, algorithm="brute", n_neighbors=1),
    "Logistic": LogisticRegression(),
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "SVM": SVC(C=0.1),
    # "Bernoulli NB": BernoulliNB()

}

def models_csv(trainX, trainy):

    names = []
    rows = []
    ntest = 10 ** 2
    if RELEASE:
        ntest = 10 ** 4
    validations = ["Exact", "One addition", "Triangular", "One missing"]

    for name, model in testModels.items():
        scores = {}
        for validation in validations:
            testX, testy = TestGenerator(trainX, trainy, type=validation)\
            .create_data(n=ntest)

            model.fit(trainX, trainy)
            score = model.score(testX, testy)
            scores[validation] = score
            names.append(name)
        mean = np.mean([score for score in scores.values()])
        mean = "%.3f" % mean
        rows.append({"Model": name, **scores, "Mean": mean})


    df = pd.DataFrame(rows, columns=["Model"] + validations + ["Mean"])
    print(df.to_csv(index=False))

def plot_models_bar(trainX, trainy):
    fig = plt.figure()
    scores = []
    names = []
    rows = []
    ntest = 10 ** 2
    if RELEASE:
        ntest = 10 ** 3
    for validation in ["Exact", "One addition", "Triangular", "One missing"]:
        testX, testy = TestGenerator(trainX, trainy, type=validation)\
            .create_data(n=ntest)
        for name, model in testModels.items():
            model.fit(trainX, trainy)
            score = model.score(testX, testy)
            scores.append(score)
            names.append(name)
            rows.append({"Model": name, "Score": score, "Validation": validation})

    df = pd.DataFrame(rows)
    ax = sns.barplot(x="Model", y="Score", hue="Validation", data=df)
    ax.set_xticklabels(df.Model.values, rotation=30)
    plt.suptitle("Prediction accuracy")
    print(df.to_csv())
    # for rect in ax.patches:
    #     height = rect.get_height()
    #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, "foo", ha='center', va='bottom')
    # for p in ax.patches:
    #     height = p.get_height()
    #     ax.text(p.get_x(), height + 3, '%1.2f' % (height))
    plt.legend()
    plt.show()


def plot_mean_bar(trainX, trainy):
    fig = plt.figure()
    rows = []
    ntest = 10 ** 2
    if RELEASE:
        ntest = 10 ** 3

    for name, model in testModels.items():
        scores = []
        for validation in ["Exact", "One addition", "Triangular", "One missing"]:
            testX, testy = TestGenerator(trainX, trainy, type=validation)\
                .create_data(n=ntest)
            model.fit(trainX, trainy)
            score = model.score(testX, testy)
            scores.append(score)
        rows.append({"Model": name, "Score": np.mean(scores)})

    rows.sort(key=lambda x: -x["Score"])
    df = pd.DataFrame(rows)
    g = sns.barplot(x="Model", y="Score", data=df, palette="Blues_d")
    plt.suptitle("Mean prediction accuracy")
    g.set_xticklabels([row["Model"] for row in rows], rotation=30)
    plt.show()

def test_regularization_params(trainX, trainy):
    ranges = np.logspace(-1, 2, num=8, base=10.0)
    for c in ranges:
        scores = []
        for validation in ["Exact", "One missing", "One addition", "Triangular"]:
            X, y = TestGenerator(trainX, trainy).create_data(n=100)
            model = SVC(C=c)
            model.fit(trainX, trainy)
            scores.append(model.score(X, y))
        yield (c, np.mean(scores))

def plot_regularization(trainX, trainy):
    keys, values = zip(*test_regularization_params(trainX, trainy))
    s = pd.Series(values, index=keys)
    s.plot()
    plt.suptitle("SVM mean accuracy with respect to C")
    plt.show()

mayo = MayoDataProvider(causeTypeFilter=CauseType.DISEASE)

def main():
    pass
if __name__ == "__main__":
    main()


