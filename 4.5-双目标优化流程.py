import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

def random_forest(max_features,ngram_range,n_estimators,max_depth):
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
    v=36000/dt
    f1=f1_score(y_val, y_pre_val, average='macro')
    return f1, v

def objective(trial):
    max_features=trial.suggest_int('max_features', 30, 100)
    ngram_range=trial.suggest_int('ngram_range', 1, 3)
    n_estimators=trial.suggest_int('n_estimators',20, 30)
    max_depth=trial.suggest_int('max_depth', 12, 30)
    f1,v=random_forest(max_features,ngram_range,n_estimators,max_depth)
    return f1,v


if __name__ == '__main__':
    storage_path = "sqlite:///storage/random_forest_demo.db"
    study = optuna.create_study(
        directions=['maximize','maximize'],
        storage=storage_path,
        load_if_exists=True,
        study_name="random_forest_demo"
    )
    study.optimize(objective, n_trials=1)
    # 打印帕累托最优解
    print("\n===== 帕累托最优解 =====")
    for trial in study.best_trials:
        print(f"Trial #{trial.number}")
        print(f"  f1: {trial.values[0]:.4f}")
        print(f"  速度(v): {trial.values[1]:.2f}")
        print("  参数:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 40)
    fig = optuna.visualization.plot_pareto_front(study, target_names=["f1", "速度"])
    fig.show()








