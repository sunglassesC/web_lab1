from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

# 词根化
ps = PorterStemmer()

posting_list_file_path = '../output/posting_list.npy'

if __name__ == '__main__':

    stop_words = set(stopwords.words('english'))

    # 词根化
    ps = PorterStemmer()

    print('开始从倒排表读数据至字典')
    posting_list = np.load(posting_list_file_path, allow_pickle=True)
    print('读取完毕')

    while 1:
        query = input('shuru')
        temp = ''
        for s in query:
            if s == '(' or s == ')':
                temp += ' ' + s + ' '
            else:
                temp += s
        expression = temp.split()

        U = set(range(5))
        for i in range(len(expression)):
            if expression[i] not in ['(', ')', 'AND', 'OR', 'NOT']:
                if expression[i] in posting_list.keys():
                    expression[i] = set(posting_list[expression[i]]['distribution'].keys())
                else:
                    expression[i] = set()
        print(expression)
