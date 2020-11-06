import os


if __name__ == '__main__':
    file_num_list = ['']        # 文件名和序号的映射
    posting_list = {}
    data_root_path = r"./dataset/test"

    seq_num = 0         # 文件名对应序号
    for root, dirs, files in os.walk(data_root_path):
        for file in files:
            seq_num += 1
            file_path = os.path.join(root, file)
            file_num_list.append(file_path)
            print(file_path)

            with open(file_path, 'r') as f:
                text = f.read()
                # 这里得到邮件的内容，需做相应处理
                # print(text)

    # 将文件名和序号的映射写入filename_map中
    with open('./output/filename_map.txt', 'w') as f:
        for ele in file_num_list:
            f.write(ele + '\n')

