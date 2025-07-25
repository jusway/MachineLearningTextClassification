import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    with open('train_corpus.pkl', 'rb') as f:
        train_corpus = pickle.load(f)

    with open('train_label.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=10,
    )
    t1=time.time()
    X_train = vectorizer.fit_transform(train_corpus)
    t2=time.time()
    print(f"训练集特征提取+转换耗时: {t2 - t1:.4f} 秒")
    print(f"特征维度: {X_train.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    classifier = RandomForestClassifier(
        n_estimators=1000,
        max_depth=200,
        n_jobs=-1,
        random_state=7
    )
    t3=time.time()
    classifier.fit(X_train, y_train)
    t4=time.time()
    print(f"训练耗时: {t4 - t3:.4f} 秒")

    t5=time.time()
    y_pre_train=classifier.predict(X_train)
    t6=time.time()
    print(f"训练集预测耗时: {t6 - t5:.4f} 秒")

    accuracy = accuracy_score(y_train, y_pre_train)
    print(f"训练集准确率: {accuracy:.4f}")
    # 验证集评估
    with open('val_corpus.pkl', 'rb') as f:
        val_corpus = pickle.load(f)
    with open('val_label.pkl', 'rb') as f:
        val_labels = pickle.load(f)
    t7=time.time()
    X_val = vectorizer.transform(val_corpus)
    y_val = label_encoder.transform(val_labels)
    t8=time.time()
    print(f"验证集特征转换耗时: {t8 - t7:.4f} 秒")
    y_pre_val = classifier.predict(X_val)
    t9=time.time()
    print(f"验证集预测耗时: {t9 - t8:.4f} 秒")
    accuracy = accuracy_score(y_val, y_pre_val)
    print(f"验证集准确率: {accuracy:.4f}")


















