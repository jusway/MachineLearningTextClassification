import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    with open('train_corpus.pkl', 'rb') as f:
        train_corpus = pickle.load(f)

    with open('train_label.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=10,
        max_features=30000,
    )
    t1=time.time()
    X = vectorizer.fit_transform(train_corpus)
    t2=time.time()
    print(f"特征提取耗时: {t2 - t1:.4f} 秒")
    print(f"特征维度: {X.shape}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_labels)

    classifier = DecisionTreeClassifier(
        max_depth=200,
        # min_samples_split=50,
        # min_samples_leaf=20,
        max_features=30000,
        # ccp_alpha=0.01,
        random_state=7
    )
    t3=time.time()
    classifier.fit(X, y)
    t4=time.time()
    print(f"训练耗时: {t4 - t3:.4f} 秒")

    t5=time.time()
    y_pre=classifier.predict(X)
    t6=time.time()
    print(f"预测耗时: {t6 - t5:.4f} 秒")

    accuracy = accuracy_score(y, y_pre)
    print(f"准确率: {accuracy:.4f}")














