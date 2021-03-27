import os

from tqdm import tqdm
import numpy as np
import math

from email.parser import Parser
from email.policy import default
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

posting_list_file_path = '../output/posting_list.npy'
tfidf_matrix_file_path = '../output/tfidf_matrix.npy'
data_root_path = "../dataset/maildir"
file_name_map_file_path = '../inf/file_name_map.txt'
first_1000_token_list_file_path = '../inf/first_1000_token_list.txt'


def tfidf_value(tf, df, N):
    if tf == 0:
        a = 0
    else:
        a = 1 + math.log10(tf)
    b = math.log10(N / df)
    return a * b


if __name__ == '__main__':

    print('开始从倒排表读数据至字典')
    posting_list = np.load(posting_list_file_path, allow_pickle=True).item()
    print('读取完毕')

    with open(first_1000_token_list_file_path, 'r') as f:
        text = f.read()
        first_1000_token_list = text.split()

    with open(file_name_map_file_path, 'r') as f:
        file_count = int(f.readline())
        file_name_map_list = f.read().split()

    # tfidf矩阵仍然是从1开始计数，对应第一个文档
    tfidf = np.ndarray(shape=(file_count + 1, 1000), dtype=np.float32)

    print('开始计算目录数...')
    directory_count = 0
    for root, dirs, files in os.walk(data_root_path):
        directory_count += 1
    print('共', directory_count, '个目录')

    print('开始建立tfidf矩阵...')
    seq_num = 0                 # 文件名对应序号
    temp_posting_list = {}      # 临时倒排表，仅记录该文档
    for file in tqdm(file_name_map_list):
        seq_num += 1  # 文件序号从1开始

        file_path = file

        with open(file_path, 'r', errors='ignore') as f:
            # 这里的步骤和建立倒排表中的一样，也需得到处理后的token
            text = f.read()

            current_email = Parser(policy=default).parsestr(text)
            body = current_email.get_payload()          # 邮件正文
            subject_text = current_email['Subject']     # 邮件主题
            body += ' ' + str(subject_text)                  # 主题并入正文，不作特殊处理

            # 分词
            word_tokens = word_tokenize(body)

            # 去停用词
            stop_words = set(stopwords.words('english'))
            filtered_tokens1 = [w for w in word_tokens if w.lower() not in stop_words]

            # 词根化
            ps = PorterStemmer()
            stemmed_tokens = [ps.stem(w) for w in filtered_tokens1 if w.isalpha()]

            # 去停用词
            stop_words = set(stopwords.words('english'))
            filtered_tokens2 = [w for w in stemmed_tokens if w.lower() not in stop_words]

            # 统计词频
            for token in filtered_tokens2:
                if token not in temp_posting_list:
                    temp_posting_list[token] = 1
                else:
                    temp_posting_list[token] += 1

            index = 0
            for token in first_1000_token_list:
                if token not in temp_posting_list.keys():
                    tfidf[seq_num][index] = 0
                else:
                    tf = temp_posting_list[token]
                    df = len(posting_list[token]['distribution'])
                    tfidf[seq_num][index] = tfidf_value(tf, df, file_count)
                index += 1

        temp_posting_list.clear()

    print('开始写入output')
    np.save(tfidf_matrix_file_path, tfidf)

