import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

def logistic_regression(max_features,ngram_range,C,solver,max_iter):
    with open('train_corpus.pkl', 'rb') as f:
        train_corpus = pickle.load(f)
    with open('train_label.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_range),
        max_df=0.8,
        min_df=10,
    )
    X_train = vectorizer.fit_transform(train_corpus)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    classifier = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=7
    )
    classifier.fit(X_train, y_train)
    # 验证集评估
    with open('val_corpus.pkl', 'rb') as f:
        val_corpus = pickle.load(f)
    with open('val_label.pkl', 'rb') as f:
        val_labels = pickle.load(f)
    y_val = label_encoder.transform(val_labels)
    t=time.time()
    X_val = vectorizer.transform(val_corpus)
    y_pre_val = classifier.predict(X_val)
    dt=time.time()-t
    v=36000/dt
    f1=f1_score(y_val, y_pre_val, average='macro')
    return f1, v

def objective(trial):
    max_features=trial.suggest_int('max_features', 1000, 50000)
    ngram_range=trial.suggest_int('ngram_range', 1, 3)
    C=trial.suggest_float('C', 1e-5, 10, log=True)
    solver=trial.suggest_categorical('solver', ['lbfgs','liblinear','newton-cg','sag','saga'])
    max_iter=trial.suggest_int('max_iter', 100, 1000)
    f1,v=logistic_regression(max_features,ngram_range,C,solver,max_iter)
    return f1,v


if __name__ == '__main__':
    storage_path = "sqlite:///storage/logistic_regression.db"
    study = optuna.create_study(
        directions=['maximize','maximize'],
        storage=storage_path,
        load_if_exists=True,
        study_name="logistic_regression"
    )
    study.optimize(objective, n_trials=200)









