import numpy as np
import math

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


posting_list_file_path = '../output/posting_list.npy'
tfidf_matrix_file_path = '../output/tfidf_matrix.npy'
first_1000_token_list_file_path = '../inf/first_1000_token_list.txt'
file_name_map_file_path = '../inf/file_name_map.txt'

def tfidf_value(tf, df, N):
    if tf == 0:
        a = 0
    else:
        a = 1 + math.log10(tf)
    b = math.log10(N / df)
    return a * b

def cosine(a, b):
    error = 10 ** (-10)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < error or norm_b < error:
        return 0
    return np.vdot(a, b) / (norm_a * norm_b)


if __name__ == '__main__':
    print('准备中...')

    posting_list = np.load(posting_list_file_path, allow_pickle=True).item()

    with open(file_name_map_file_path, 'r') as f:
        file_count = int(f.readline())
        file_name_map = f.read().split()

    tfidf = np.load(tfidf_matrix_file_path, allow_pickle=True)

    with open(first_1000_token_list_file_path, 'r') as f:
        text = f.read()
        first_1000_token_list = text.split()

    ps = PorterStemmer()
    stopwords_list = stopwords.words('english')
    while 1:
        query = input('输入查询，将返回相关度前10的文档路径，输入"$quit"将退出程序\n')
        if query == '$quit':
            break

        word_list = query.split()

        # 对于查询词也需做词根化和去停用词处理
        stemmed_tokens = [ps.stem(w) for w in word_list if w.isalpha()]
        word_list = [w for w in stemmed_tokens if w.lower() not in stopwords_list]

        # 构建查询向量，出现的词为1，未出现的为0
        query_vec = np.zeros(1000)
        for word in word_list:
            try:
                index = first_1000_token_list.index(word)
            #    print(index)
                tf = 1
                df = len(posting_list[word]['distribution'])
                query_vec[index] = tfidf_value(tf, df, file_count)
            #    print(tfidf_value(tf,df,file_count))
            except:
                print('查询词: ' + word + '不在查询词表')

        length = tfidf.shape[0]
        temp = [0.0]
        for i in range(1, length):
            temp.append(cosine(tfidf[i], query_vec))

        result = []
        for i in range(10):
            index = temp.index(max(temp))
            print('余弦相似度为： ',max(temp))
            result.append(index)
            temp[index] = 0

        count = 1
        for ele in result:
            print(str(count), ': ', file_name_map[ele-1])
            count += 1
