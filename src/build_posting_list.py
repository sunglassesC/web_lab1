import os
from email.parser import Parser
from email.policy import default

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import numpy as np

from tqdm import tqdm

data_root_path = "../dataset/maildir"
posting_list_file_path = '../output/posting_list.npy'
file_name_map_file_path = '../inf/file_name_map.txt'
first_1000_token_list_file_path = '../inf/first_1000_token_list.txt'

if __name__ == '__main__':
    file_num_list = ['']    # 文件名和序号的映射，0号元素为一个空的字符串，即文件编号从1开始
    posting_list = {}       # 倒排表

    print('开始计算目录数...')
    directory_count = 0
    for root, dirs, files in os.walk(data_root_path):
        directory_count += 1
    print('共', directory_count, '个目录')

    print('开始建立倒排表...')
    seq_num = 0  # 文件名对应序号
    for root, dirs, files in tqdm(os.walk(data_root_path), total=directory_count):
        for file in files:
            seq_num += 1        # 文件序号从1开始

            file_path = os.path.join(root, file)
            file_num_list.append(file_path)

            with open(file_path, 'r', errors='ignore') as f:
                # 这里得到邮件的内容，需做相应处理
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

                # 统计词频和分布
                for token in filtered_tokens2:
                    if token not in posting_list:
                        posting_list[token] = {
                            'freq': 1,                      # 词频
                            'distribution': {seq_num: 1}    # 分布，格式为  文档编号: 出现次数
                        }
                    else:
                        posting_list[token]['freq'] += 1
                        if seq_num not in posting_list[token]['distribution']:
                            posting_list[token]['distribution'][seq_num] = 1
                        else:
                            posting_list[token]['distribution'][seq_num] += 1

    first_1000_token = {}
    for i in range(1000):
        if posting_list:
            a = max(posting_list.items(), key=lambda x: x[1]['freq'])
            first_1000_token[a[0]] = a[1]
            del posting_list[a[0]]
        else:
            print('list is already empty')
            break

    # print(first_1000_token)
    print('开始写入output')
    # 将词频前1000的词的倒排表写入文件
    np.save(posting_list_file_path, first_1000_token)

    # 将这1000个词按顺序写入文件(原因是python字典是无序的，后续tfidf矩阵的处理需要词的顺序)
    with open(first_1000_token_list_file_path, 'w') as f:
        for ele in first_1000_token.keys():
            f.write(ele + '\n')

    # 将文件名和序号的映射写入filename_map中
    file_num_list[0] = str(seq_num)                 # 0号元素写入文件总数
    with open(file_name_map_file_path, 'w') as f:
        for ele in file_num_list:
            f.write(ele + '\n')
