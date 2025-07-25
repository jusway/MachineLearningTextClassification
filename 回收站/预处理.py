import time
import pickle
from utils import load_jsonl_data
from utils import cut_text

t1 = time.time()
texts, labels = load_jsonl_data('val.json')
corpus = []
for text in texts:
    text_processed = cut_text(text)
    corpus.append(text_processed)
print(corpus[:10])
t2 = time.time()
print(f"文本预处理耗时: {t2 - t1:.4f} 秒")
with open('../val_corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('../val_label.pkl', 'wb') as f:
    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)