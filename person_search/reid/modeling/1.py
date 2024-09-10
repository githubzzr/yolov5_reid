import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print("当前文件目录:", current_dir)

# 计算项目根目录的绝对路径
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
print("项目根目录:", root_dir)
file_path = os.path.join(root_dir, "weights")
print("项目根目录:", file_path)