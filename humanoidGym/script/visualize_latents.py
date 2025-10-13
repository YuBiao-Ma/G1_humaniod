import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
plt.rcParams['pdf.fonttype'] = 42   # 使用 TrueType，非 Type 3
plt.rcParams['ps.fonttype'] = 42
# 无显示环境或服务器上运行时：强制使用无界面后端
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")          # ★ 在 import pyplot 之前
# 不要工具栏，避免 Tk 创建图标
matplotlib.rcParams['toolbar'] = 'None'
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

# 读取数据
data = pd.read_csv('random_uniform_terrain.csv')
pred_classes = data.iloc[:, -1].values

# 统计每个类别的出现次数和百分比
unique_classes, counts = np.unique(pred_classes, return_counts=True)
percentages = (counts / len(pred_classes)) * 100

# 只保留出现次数最高的5个类别
if len(unique_classes) > 5:
    # 获取前5个最大值的索引
    top5_indices = np.argsort(counts)[-5:][::-1]
    unique_classes = unique_classes[top5_indices]
    counts = counts[top5_indices]
    percentages = percentages[top5_indices]

# 创建颜色映射
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

# 创建柱状图
plt.figure(figsize=(10, 8))
bars = plt.bar(range(len(unique_classes)), counts, color=colors, edgecolor='black', alpha=0.8)

# 设置图形属性
plt.xlabel('Dictionary Index', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.grid(True, alpha=0.3, axis='y')

# 设置x轴标签为类别名称
plt.xticks(range(len(unique_classes)), unique_classes)

# 在每个柱子上方只显示百分比
for i, (bar, percentage) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    plt.text(i, height + max(counts)*0.01,
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()

# 保存为PDF文件
plt.savefig('random_uniform_terrain.pdf', format='pdf', bbox_inches='tight')

# 显示图形（可选）
plt.show()