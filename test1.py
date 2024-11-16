import numpy as np

# 加载 npz 文件
data = np.load('mnist.npz')

# 列出文件中的所有数组
print(data.files)

# 读取特定数组
array1 = data['array_name1']
array2 = data['array_name2']

# 打印数组内容
print(array1)
print(array2)
