# import wandb
# import os

# # 设置环境变量以使用离线模式
# os.environ['WANDB_MODE'] = 'dryrun'

# # 初始化运行，指定目录并允许恢复
# run = wandb.init(dir='/home/chenyupeng/wandb/run-20241108_163240-1axdm5yr', resume='allow')

# # 访问历史数据
# history = run.history()
# print("历史数据：")
# print(history)

# # 访问配置数据
# config = run.config
# print("\n配置数据：")
# print(dict(config))

# # 访问摘要数据
# summary = run.summary
# print("\n摘要数据：")
# print(dict(summary))

# # 结束运行
# wandb.finish()


import os
import pandas as pd

run_dir = '/home/chenyupeng/wandb/run-20241108_163240-1axdm5yr'

# 加载历史数据
history_file = os.path.join(run_dir, 'history.csv')
if os.path.exists(history_file):
    history = pd.read_csv(history_file)
    print("历史数据（history.csv）：")
    print(history)
else:
    print("在运行目录中未找到 history.csv 文件。")
