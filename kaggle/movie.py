import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

output_path = 'output/movie.pkl'
file_path = './input/movie-review-sentiment-analysis-kernels-only/train.tsv'

df = pd.read_csv(file_path, delimiter='\t')
df['len'] = df['Phrase'].map(lambda x: len(x.split(' ')))
df = df[df.len > 10]


print('create vector')
vec_tfidf = TfidfVectorizer(max_df=0.3, max_features=1000)
X = vec_tfidf.fit_transform(df['Phrase'])
vec_df = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(vec_df, df['Sentiment'], test_size=0.33, random_state=42)

if not os.path.isfile(output_path):
    print('create model')

    # 線形SVMのインスタンスを生成
    model = SVC(kernel='rbf', random_state=None, verbose=1)
    # モデルの学習。fit関数で行う。
    model.fit(X_train[:5000], y_train[:5000])

    pred = model.predict(X_test)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
else:
    print('load model')
    with open(output_path, 'rb') as f:
        model = pickle.load(f)
        pred = model.predict(X_test[:100])
        print(accuracy_score(pred, y_test[:100]))
