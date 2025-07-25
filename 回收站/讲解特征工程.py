from sklearn.feature_extraction.text import CountVectorizer

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# corpus2=[
#     "are you ok",
#     "is",
#     "is first"
# ]
# vectorizer = CountVectorizer()
# fit = vectorizer.fit(corpus)
# transform = fit.transform(corpus2)
#
# print(type(fit))
# print(fit.get_feature_names_out())
# print(type(transform))
# print(transform.toarray())

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
corpus2=[
    "are you ok",
]

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names_out())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 1))
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(X2.toarray())

















