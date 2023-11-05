import matplotlib.pyplot as plt
import os

# 设置更大的图形尺寸和分辨率
plt.figure(figsize=(10, 6), dpi=100)

file_name = 'ricogpt_rn50_0415'
log_path = '/root/autodl-tmp/UI_ITC/logs/'
full_file_path = os.path.join(log_path, file_name + '.log')

epoch_id = []
training_loss = []
evaluation_loss = []

# 使用with语句安全打开文件
with open(full_file_path, 'r') as file:
    for row in file:
        print(row)
        if 'Namespace' in row:
            continue

        row = row.split(': ')
        if 'train_epoch' in row[0]:
            training_loss.append(float(row[1]))
        elif 'eva' in row[0]:
            evaluation_loss.append(float(row[1]))
        elif 'epoch ' == row[0]:
            epoch_id.append(int(row[1]))
        else:
            continue

# 绘制曲线，添加线宽和线型以及标记，以提高可读性
plt.plot(epoch_id, training_loss, label='Training Loss', linewidth=2, marker='o', linestyle='-', markersize=5)
plt.plot(epoch_id, evaluation_loss, label='Evaluation Loss', linewidth=2, marker='x', linestyle='--', markersize=5)

# 添加轴标签和标题
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Evaluation Loss Over Epochs', fontsize=16, pad=20)

# 添加图例
plt.legend(fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局
plt.tight_layout()

# 保存图像
output_file_path = os.path.join(log_path, file_name + '.png')  # 改为.png，因为它是一个图像格式
plt.savefig(output_file_path)

# 显示图像（如果在notebook等环境中运行）
plt.show()
