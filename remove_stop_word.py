# Tạo file stopword
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(txt_file):
    """
    Each document is one line, documents is already preprocess like: remove truncate, tokenize, strip, ...
    :param txt_file: path/to/text/file
    :return: list of documents
    """
    texts = []
    with open(txt_file, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            texts.append(line.strip())
    return texts


def get_stopwords(documents, threshold=3):
    """
    :param documents: list of documents
    :param threshold:
    :return: list of words has idf <= threshold
    """
    tfidf = TfidfVectorizer(min_df=100)
    tfidf_matrix = tfidf.fit_transform(documents)
    features = tfidf.get_feature_names()
    stopwords = []
    print(min(tfidf.idf_), max(tfidf.idf_), len(features))
    for index, feature in enumerate(features):
        if tfidf.idf_[index] <= threshold:
            stopwords.append(feature)
    return stopwords


if __name__ == '__main__':
    docs = load_data(r"D:\Users\Admin\Desktop\corpus-full.pre")
    stopwords = get_stopwords(docs, threshold=3)
    with open('stopwords.txt', 'w', encoding='utf8') as fp:
        for word in stopwords:
            fp.write(word + '\n')




# Loại bỏ stopwords 
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopwords:
            words.append(word)
    return ' '.join(words)

def get_stop_word():
    stopwords = []
    stopwordFile = open('data/vietnamese-stopwords.txt', 'r')
    for stopword in stopwordFile:
        stopwords.append(stopword.strip())
    
    return stopwords

stopwords = get_stop_word()
sentence = "đây là một sản phẩm rất là tốt"

print("tốt" in stopwords)
