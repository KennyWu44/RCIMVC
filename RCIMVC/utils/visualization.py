import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


import torch
import seaborn as sns


colorList = [
    'darkblue',
    'darkviolet',
    'red',
    'deeppink',
    'darkslategray',
    'darkgreen',
    'fuchsia',
    'crimson',
    'aqua',
    'steelblue',
    'dodgerblue',
    'royalblue',
    'greenyellow',
    'lightblue',
    'lightgreen',
    'slateblue',
    'orange',
    'orangered',
    'yellow',
    'gold',
    'sandybrown',
    'gray',
]


def TSNE_show2D(z, y):
    """use t-SNE to visualize the latent representation"""
    # 如果 z 是一个 PyTorch Tensor，且在 GPU 上，先将其转移到 CPU 并转换为 NumPy 数组
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    t_sne = TSNE(n_components=2, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]

    plt.scatter(data[0], data[1], c=color, marker='.', s=12)
    plt.axis('off')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()


# def TSNE_show3D(z, y):
#     """use t-SNE to visualize 3D the latent representation"""
#     t_sne = TSNE(n_components=3, learning_rate='auto')
#     data = t_sne.fit_transform(z)
#     data = pd.DataFrame(data, index=y)
#     color = [colorList[i - 1] for i in data.index]

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[0], data[1], data[2], c=color, s=12)

#     ax.set_xlim3d(min(data[0]), max(data[0]))
#     ax.set_ylim3d(min(data[1]), max(data[1]))
#     ax.set_zlim3d(min(data[2]), max(data[2]))

#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])

#     plt.tick_params(labelsize=8)
#     # plt.axis('off')
#     ax.grid(True)
#     # ax.view_init(10, 185)
#     plt.subplots_adjust(left=-0., right=1., top=1., bottom=-0.)
#     plt.tight_layout()
#     plt.savefig('../imgs/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
#     plt.show()


def TSNE_show3D(z, y):
    """use t-SNE to visualize 3D the latent representation"""
    # 如果 z 是一个 PyTorch Tensor，且在 GPU 上，先将其转移到 CPU 并转换为 NumPy 数组
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()

    t_sne = TSNE(n_components=3, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2], c=color, s=12)

    ax.set_xlim3d(min(data[0]), max(data[0]))
    ax.set_ylim3d(min(data[1]), max(data[1]))
    ax.set_zlim3d(min(data[2]), max(data[2]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tick_params(labelsize=8)
    ax.grid(True)
    plt.subplots_adjust(left=-0., right=1., top=1., bottom=-0.)
    plt.tight_layout()
    plt.savefig('' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()




def loss_plot(loss, acc, nmi, ari, print_num):
    epochs_loss = range(1, len(loss) + 1)  # 对应 LOSS 的 epoch 数
    epochs_eval = range(print_num, len(loss) + 1, print_num)  # 评估的 epoch 数

    fig = plt.figure(figsize=(8, 5))
    ax_left = fig.add_subplot(111)
    ax_right = ax_left.twinx()

    ax_left.set_xlabel('Epoch', fontsize=14)
    ax_left.set_ylabel('Clustering Performance', fontsize=14)
    ax_right.set_ylabel('Loss', fontsize=14)

    # 绘制 Loss 曲线
    a1 = ax_right.plot(epochs_loss, loss, color='#F27970', label='Loss')
    # 绘制 acc, nmi, ari 曲线 (只在评估点绘制)
    a2 = ax_left.plot(epochs_eval, acc, color='#54B345', label='ACC')
    a3 = ax_left.plot(epochs_eval, nmi, color='#05B9E2', label='NMI')
    a4 = ax_left.plot(epochs_eval, ari, color='#BB9727', label='ARI')

    # 合并图例
    lns = a1 + a2 + a3 + a4
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right')

    plt.tight_layout()
    plt.savefig('loss.png')
    plt.show()

# # 调用时传入 print_num 参数
# loss_plot(LOSS, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ARI'], print_num)

# def loss_plot(loss, acc, nmi, ari):
#     epochs = range(1, len(loss) + 1)
#     fig = plt.figure(figsize=(8, 5))
#     ax_left = fig.add_subplot(111)
#     ax_right = ax_left.twinx()

#     ax_left.set_xlabel('Epoch', fontsize=14)
#     ax_left.set_ylabel('Clustering Performance', fontsize=14)
#     ax_right.set_ylabel('Loss', fontsize=14)

#     a1 = ax_right.plot(epochs, loss, color='#F27970', label='Loss')
#     a2 = ax_left.plot(epochs, acc, color='#54B345', label='ACC')
#     a3 = ax_left.plot(epochs, nmi, color='#05B9E2', label='NMI')
#     a4 = ax_left.plot(epochs, ari, color='#BB9727', label='ARI')

#     lns = a1 + a2 + a3 + a4
#     labs = [l.get_label() for l in lns]
#     ax_left.legend(lns, labs, loc='center right')
#     # ax_right.legend(loc='lower right', fontsize=10)
#     # ax_left.legend(loc='center right', fontsize=10)

#     plt.tight_layout()
#     plt.savefig('loss.pdf')
#     plt.show()



def miss_plot(flag=1):
    csv_dir='../data/'
    
    # 设置缺失率和准确率的均值和标准差
    missing_rates = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])

    data_name = None
    if flag == 1:
        data_name = 'acc'
    elif flag == 2:
        data_name = 'nmi'
    elif flag == 3:
        data_name = 'ari'

    # 读取数据
    data = pd.read_csv(csv_dir + data_name + '.csv')
    data = np.array(data)

    # 获取每个算法的均值和标准差
    BSV_mean, BSV_std = data[:, 0], data[:, 1]
    Concat_mean, Concat_std = data[:, 2], data[:, 3]
    PVC_mean, PVC_std = data[:, 4], data[:, 5]
    MIC_mean, MIC_std = data[:, 6], data[:, 7]
    DAIMC_mean, DAIMC_std = data[:, 8], data[:, 9]
    Completer_mean, Completer_std = data[:, 10], data[:, 11]
    DSIMVC_mean, DSIMVC_std = data[:, 12], data[:, 13]
    DIMVC_mean, DIMVC_std = data[:, 14], data[:, 15]
    Ours_mean, Ours_std = data[:, 16], data[:, 17]

    # 自定义颜色
    colors = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#2878b5', '#c82423']

    plt.figure(figsize=(8, 5))
    # 绘制误差带图
    plt.plot(missing_rates, BSV_mean, color=colors[0], label='BSV')
    plt.fill_between(missing_rates, BSV_mean - BSV_std, BSV_mean + BSV_std, alpha=0.2, color=colors[0])

    plt.plot(missing_rates, Concat_mean, color=colors[1], label='Concat')
    plt.fill_between(missing_rates, Concat_mean - Concat_std, Concat_mean + Concat_std, alpha=0.2, color=colors[1])

    plt.plot(missing_rates, PVC_mean, color=colors[2], label='PVC')
    plt.fill_between(missing_rates, PVC_mean - PVC_std, PVC_mean + PVC_std, alpha=0.2, color=colors[2])

    plt.plot(missing_rates, MIC_mean, color=colors[3], label='MIC')
    plt.fill_between(missing_rates, MIC_mean - MIC_std, MIC_mean + MIC_std, alpha=0.2, color=colors[3])

    plt.plot(missing_rates, DAIMC_mean, color=colors[4], label='DAIMC')
    plt.fill_between(missing_rates, DAIMC_mean - DAIMC_std, DAIMC_mean + DAIMC_std, alpha=0.2, color=colors[4])

    plt.plot(missing_rates, Completer_mean, color=colors[5], label='Completer')
    plt.fill_between(missing_rates, Completer_mean - Completer_std, Completer_mean + Completer_std, alpha=0.2,
                     color=colors[5])

    plt.plot(missing_rates, DSIMVC_mean, color=colors[6], label='DSIMVC')
    plt.fill_between(missing_rates, DSIMVC_mean - DSIMVC_std, DSIMVC_mean + DSIMVC_std, alpha=0.2, color=colors[6])

    plt.plot(missing_rates, DIMVC_mean, color=colors[7], label='DIMVC')
    plt.fill_between(missing_rates, DIMVC_mean - DIMVC_std, DIMVC_mean + DIMVC_std, alpha=0.2, color=colors[7])

    plt.plot(missing_rates, Ours_mean, color=colors[8], label='Ours')
    plt.fill_between(missing_rates, Ours_mean - Ours_std, Ours_mean + Ours_std, alpha=0.2, color=colors[8])

    plt.xlabel('Missing rate', fontsize=22)
    if flag == 1:
        plt.ylabel('Accuracy (%)', fontsize=22)
        plt.title('Accuracy with different missing rates', fontsize=22)
    elif flag == 2:
        plt.ylabel('NMI (%)', fontsize=22)
        plt.title('NMI with different missing rates', fontsize=22)
    elif flag == 3:
        plt.ylabel('ARI (%)', fontsize=22)
        plt.title('ARI with different missing rates', fontsize=18)

    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=12)
    # plt.xticks(missing_rates)

    # 保存为PDF格式并边距极小
    plt.tight_layout()
    plt.savefig('../imgs/' + data_name + '.pdf')


def k_anl_plot():
    csv_dir='../data/'
    # 读取数据
    data = pd.read_csv(csv_dir+'K.csv')
    data = np.array(data)
    K = range(1, len(data[:, 0]) + 1)

    plt.figure(figsize=(8, 5))

    # 自定义颜色
    colors = ['#54B345', '#05B9E2', '#c82423']
    plt.plot(K, data[:, 0], color=colors[0], label='ACC', marker='o')
    plt.plot(K, data[:, 1], color=colors[1], label='NMI', marker='+')
    plt.plot(K, data[:, 2], color=colors[2], label='ARI', marker='*')

    plt.xlabel('Number of K', fontsize=16)
    plt.ylabel('Clustering Performance (%)', fontsize=16)
    # plt.title('ARI with different missing rates', fontsize=18)

    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=12)
    plt.xticks(K[::2])

    # 保存为PDF格式并边距极小
    plt.tight_layout()
    plt.savefig('../imgs/K_anl.pdf')


# if __name__ == '__main__':
#     miss_plot(1)
#     miss_plot(2)
#     miss_plot(3)

# k_anl_plot()



def plot_sparse_tensor_heatmap(sparse_tensor, cmap="YlGn", save_path=None, show_color_bar=False):
    """
    将稀疏张量转换为密集张量，并绘制热力图。
    
    参数:
    - sparse_tensor: 需要绘制的稀疏张量 (torch.sparse_coo)
    - cmap: 热力图的颜色映射，默认为 "YlGn"
    - save_path: 可选，保存图像的路径，若为 None 则不保存
    - show_color_bar: 是否显示 color bar，默认为 False
    
    返回:
    - None
    """
    # 检查是否为稀疏张量
    if sparse_tensor.is_sparse:
        # 转换为密集张量并移动到 CPU
        dense_tensor = sparse_tensor.to_dense().cpu().numpy()
    else:
        raise ValueError("输入的张量必须是稀疏张量")

    # 创建一个新图像
    plt.figure(figsize=(8, 8))
    
    # 使用 seaborn 绘制热力图
    ax = sns.heatmap(dense_tensor, cmap=cmap, 
                     square=True,  # 正方形格子
                     cbar=show_color_bar,  # 是否显示 color bar
                     xticklabels=False, yticklabels=False)  # 去除纵、横轴 label
    
    # 如果指定了保存路径，则保存图像
    if save_path:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.0)  # 去白边
        plt.close(fig)  # 关闭图像以释放内存
    else:
        plt.show()  # 如果未指定保存路径，则显示图像


from sklearn.metrics.pairwise import cosine_similarity

def visualize_cosine_similarity(matrices, titles=None, save_path=None):
    """
    计算并可视化多个潜在表示矩阵的余弦相似度矩阵，支持 PyTorch CUDA 张量。
    
    参数:
    - matrices: 包含潜在表示矩阵的列表，每个矩阵可以是 NumPy 数组或 PyTorch 张量。
    - titles: 每个相似度矩阵对应的标题列表，长度应与 matrices 相同。如果为 None，将自动命名为 Matrix 1, Matrix 2, 等。
    - save_path: 如果提供了保存路径，则将图像保存到该路径，否则显示图像。
    """
    num_matrices = len(matrices)
    
    # 设置默认标题
    if titles is None:
        titles = [f'Matrix {i+1}' for i in range(num_matrices)]
    
    # 创建子图布局
    fig, axes = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))
    
    if num_matrices == 1:
        axes = [axes]  # 如果只有一个矩阵，确保 axes 是列表形式
    
    for i, matrix in enumerate(matrices):
        # 如果是 PyTorch 张量，先将其转为 CPU 并转换为 NumPy 数组
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(matrix)
        
        # 绘制热力图，设置vmin和vmax
        sns.heatmap(similarity_matrix, ax=axes[i], cmap='viridis', annot=False, vmin=0, vmax=1)
        axes[i].set_title(titles[i])
    
    plt.tight_layout()
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()

# # 示例使用 (包含 CUDA 张量)
# H = torch.rand(10, 5).cuda()  # PyTorch CUDA tensor
# H1 = torch.rand(10, 5)        # PyTorch CPU tensor
# H2 = np.random.rand(10, 5)    # NumPy array

# matrices = [H, H1, H2]
# titles = ['H Similarity', 'H1 Similarity', 'H2 Similarity']

# visualize_cosine_similarity(matrices, titles, save_path='cosine_similarity_adjusted.png')
