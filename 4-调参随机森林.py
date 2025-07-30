import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

def random_forest(max_features,ngram_range,max_df,min_df,n_estimators,max_depth):
    with open('train_corpus.pkl', 'rb') as f:
        train_corpus = pickle.load(f)
    with open('train_label.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_range),
        max_df=max_df,
        min_df=min_df,
    )
    X_train = vectorizer.fit_transform(train_corpus)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
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
    dt=dt/60
    f1=f1_score(y_val, y_pre_val, average='macro')
    print(f"f1 {f1}    dt {dt}")
    return f1/ dt

def objective(trial):
    max_features=trial.suggest_int('max_features', 1000, 50000)
    ngram_range=trial.suggest_int('ngram_range', 1, 3)
    max_df=trial.suggest_float('max_df', 0.1, 1.0)
    min_df=trial.suggest_int('min_df', 1, 100)
    n_estimators=trial.suggest_int('n_estimators',50, 5000)
    max_depth=trial.suggest_int('max_depth', 2, 100)
    y=random_forest(max_features,ngram_range,max_df,min_df,n_estimators,max_depth)
    return y


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)