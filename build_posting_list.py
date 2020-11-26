import os
from email.parser import Parser
from email.policy import default
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm


if __name__ == '__main__':
    file_num_list = ['']    # 文件名和序号的映射，0号元素为一个空的字符串，即文件编号从1开始
    posting_list = {}       # 倒排表
    data_root_path = r"./dataset/maildir"

    print('开始计算目录数...')
    files_count = 0
    for root, dirs, files in os.walk(data_root_path):
        files_count += 1
    print('共', files_count, '个目录')

    print('开始建立倒排表...')
    seq_num = 0  # 文件名对应序号
    for root, dirs, files in tqdm(os.walk(data_root_path), total=files_count):
        for file in files:
            seq_num += 1        # 文件序号从1开始

            file_path = os.path.join(root, file)
            file_num_list.append(file_path)

            with open(file_path, 'r', errors='ignore') as f:
                # 这里得到邮件的内容，需做相应处理
                try:
                    text = f.read()
                except UnicodeDecodeError:
                    print(file_path)
                else:
                    current_email = Parser(policy=default).parsestr(text)
                    body = current_email.get_payload()          # 邮件正文
                    subject_text = current_email['Subject']     # 邮件主题
                    body += ' ' + subject_text                  # 主题并入正文，不作特殊处理

                    # 分词
                    word_tokens = word_tokenize(body)

                    # 词根化
                    ps = PorterStemmer()
                    stemmed_tokens = [ps.stem(w) for w in word_tokens if w.isalpha()]

                    # 去停用词
                    stop_words = set(stopwords.words('english'))
                    filtered_tokens = [w for w in stemmed_tokens if w.lower() not in stop_words]

                    for token in filtered_tokens:
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
    with open('./output/postinglist.txt', 'w') as f:
        f.write(json.dumps(first_1000_token))

    # 将文件名和序号的映射写入filename_map中
    with open('./output/filename_map.txt', 'w') as f:
        for ele in file_num_list:
            f.write(ele + '\n')
