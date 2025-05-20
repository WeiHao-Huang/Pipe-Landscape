# 导入函数
from data.slimpajama import get_slimpajama_data

# 定义数据集路径
datasets_dir = "/mntcephfs/lab_data/chenyupeng/llm_datasets"

# 调用函数并打印结果
data_paths = get_slimpajama_data(datasets_dir)
print(data_paths)