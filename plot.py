import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager
font_dirs = ['/usr/share/fonts', 'C:/Windows/Fonts']
fonts = font_manager.findSystemFonts(fontpaths=font_dirs)
print(fonts)
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 创建一个有向图
G = nx.DiGraph()

# 添加节点
nodes = [
    ("语义提取", "提取特征维度"),
    ("语义重要性评估", "评估语义重要性"),
    ("动态调度与交织策略调整", "为不同特征分配资源"),
    ("编码与调制", "根据交织策略"),
    ("传输与接收", "通过信道传输"),
    ("错误纠正与修复", "解交织、修复损坏信息")
]

# 添加节点名称和标签
for node, label in nodes:
    G.add_node(node, label=label)

# 添加边 (流程的顺序)
edges = [
    ("语义提取", "语义重要性评估"),
    ("语义重要性评估", "动态调度与交织策略调整"),
    ("动态调度与交织策略调整", "编码与调制"),
    ("编码与调制", "传输与接收"),
    ("传输与接收", "错误纠正与修复")
]
G.add_edges_from(edges)

# 设置图形的布局
pos = {
    "语义提取": (0, 1),
    "语义重要性评估": (1, 1),
    "动态调度与交织策略调整": (2, 1),
    "编码与调制": (3, 1),
    "传输与接收": (4, 1),
    "错误纠正与修复": (5, 1)
}

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制节点和边
nx.draw(G, pos, with_labels=True, node_size=4000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)

# 添加标签
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="black")

# 设置标题
plt.title("基于语义重要性的动态语义交织技术流程图", fontsize=14)

# 显示图形
plt.show()
plt.savefig("plot.png")
