import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, matthews_corrcoef

np.random.seed(2021)
def POS(X,y,p=1,n_samples=100):
    n_feature = X.shape[1]
    # 0 negative, 1 positive
    X_pos = X[y==1]
    X_neg = X[y==0]
    n_pos, n_neg = X_pos.shape[0],X_neg.shape[0]
    # n_samples = n_neg - n_pos
    new_X = []
    for i in range(n_samples):
        seed_idx = np.random.choice(n_pos)
        seed = X_pos[seed_idx]
        perturbation = np.random.multivariate_normal(np.zeros((n_feature)),np.eye(n_feature)/n_pos**p)
        num_subspace = np.random.randint(0,n_feature)
        mask_idx = np.random.permutation(n_feature)[:num_subspace]
        perturbation[mask_idx] = 0
        new_X.append(seed + perturbation)
    new_X = np.array(new_X)
    new_y = np.ones(n_samples,)
    X_res = np.vstack((X,new_X))
    y_res = np.hstack((y,new_y))
    return X_res,y_res

X, y = make_moons(5000,noise=0.2)
X_pos, X_neg = X[y==1], X[y==0]
y_pos, y_neg = y[y==1], y[y==0]
rand_idx = np.random.permutation(X_pos.shape[0])[:100]
X_pos, y_pos = X_pos[rand_idx], y_pos[rand_idx]
X = np.vstack((X_pos,X_neg))
y = np.hstack((y_pos,y_neg))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

clf = GradientBoostingClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print(matthews_corrcoef(y_test,pred))

X_res, y_res = POS(X_train,y_train,p=1)
clf = GradientBoostingClassifier()
clf.fit(X_res,y_res)
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print(matthews_corrcoef(y_test,pred))

