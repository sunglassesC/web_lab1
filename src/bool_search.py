from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
### NOT > AND > OR 优先级

posting_list_file_path = '../output/posting_list.npy'
file_name_map_file_path = '../inf/file_name_map.txt'

operator = ['NOT', 'AND', 'OR', '(', ')']    # 布尔运算符
operator_high = ['NOT', '(']     # 优先级最高的运算符
operator_middle = ['NOT', 'AND'] # 优先级高于OR的运算符

def middle2behind(expresssion):     # 将正常顺序的查询条件转换为后缀表达方式的函数
    result = []             # 转换结果列表
    stack = []              # 栈
    for item in expression:
        if item not in operator:      # 如果当前是查询词，那么直接放入结果列表
            result.append(item)
        else:                     # 如果当前为一切其他操作符
            if len(stack) == 0:   # 如果栈空，直接入栈
                stack.append(item)
            elif item in operator_high:   # 如果当前为NOT(，直接入栈
                stack.append(item)
            elif item == ')':     # 如果右括号则全部弹出（碰到左括号停止）
                t = stack.pop()
                while t != '(':   
                    result.append(t)
                    t = stack.pop()
            # 如果当前词为AND，且栈顶为NOT，则开始弹出
            elif item in 'AND' and stack[len(stack)-1] in 'NOT':
                if stack.count('(') == 0:           # 如果没有左括号，弹出所有     
                    while stack:
                        result.append(stack.pop())
                else:                               # 如果有左括号，弹到左括号为止 
                    t = stack.pop()
                    while t != '(':
                        result.append(t)
                        t = stack.pop()
                    stack.append('(')
                stack.append(item)  # 弹出操作完成后将‘AND’入栈
            elif item in 'OR' and stack[len(stack)-1] in operator_middle:
                if stack.count('(') == 0:
                    while stack:
                        result.append(stack.pop())
                else:
                    t = stack.pop()
                    while t != '(':
                        result.append(t)
                        t = stack.pop()
                    stack.append('(')
                stack.append(item) # 将‘OR’入栈
            else:
                stack.append(item)# 其余情况直接入栈

    # 表达式遍历完了，但是栈中还有操作符不满足弹出条件，把栈中的东西全部弹出
    while stack:
        result.append(stack.pop())
    # 返回结果
    return result

if __name__ == '__main__':
    print("准备中...")
    posting_list = np.load(posting_list_file_path, allow_pickle=True).item()

    with open(file_name_map_file_path, 'r') as f:
        files_num = int(f.readline())   # files_num 为全部文件名的个数
        file_name_map = f.read().split()    # 文件名

    while 1:
        query = input('输入布尔查询条件，将返回符合条件的文档路径，输入"$quit"将退出程序\n')  # 输入查询条件
        if query == '$quit':
            break

        expression = word_tokenize(query)   # 对查询条件进行分词处理
        print('Query after tokenize:  ', expression)

#print('After changing:  ', middle2behind(expression))

        expression_result = middle2behind(expression)
        print('After changing:  ', expression_result)

        stemmed_expression = []     # 对后缀表达方式的查询语句进行词根化
        for word in expression_result:
            if word.isalpha():
                stemmed_expression.append(PorterStemmer().stem(word))
        print('After stemming:  ', stemmed_expression)

        dictionary = posting_list
        all_files_temp = []

        for i in range(files_num):
            t = i + 1
            all_files_temp.append(t)
        all_files = set(all_files_temp)     # 转换为集合，表示所有文件名的集合

        for i in range(len(stemmed_expression)):
            if stemmed_expression[i] not in ['not', 'and', 'OR']:
                if stemmed_expression[i] not in dictionary.keys():
                    print('查询词: ' + stemmed_expression[i] + '不在查询词表')
                    stemmed_expression[i] = set()
                else:
                    stemmed_expression[i] = set(dictionary[stemmed_expression[i]]['distribution'].keys())
            elif stemmed_expression[i] == 'not':
                stemmed_expression[i] = {'-'}
            elif stemmed_expression[i] == 'and':
                stemmed_expression[i] = {'&'}
            elif stemmed_expression[i] == 'OR':
                stemmed_expression[i] = {'|'}
        # print('After replacing:  ', stemmed_expression)

        operators = [{'-'}, {'&'}, {'|'}]
        answer = []

        for item in stemmed_expression:
            if item not in operators:
                answer.append(item)
            elif item == {'-'}:
                op1 = answer.pop()
                answer.append(all_files - op1)
            elif item == {'&'}:
                op1 = answer.pop()
                op2 = answer.pop()
                answer.append(op1 & op2)
            elif item == {'|'}:
                op1 = answer.pop()
                op2 = answer.pop()
                answer.append(op1 | op2)

        count = 1
        for ele in answer.pop():
            return_answer = str(count) + ': ' + file_name_map[ele-1]
            print(return_answer)
            # print(str(count), ': ', file_name_map[ele])
            count += 1
        #    doc = open('write_answer.txt', 'a')
        #    print(return_answer, file=doc)
# print("The answer for the query is:  ", answer)
