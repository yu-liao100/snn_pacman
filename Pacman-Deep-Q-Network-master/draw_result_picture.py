import matplotlib.pyplot as plt
import re

from rl_plotter.logger import Logger
# 读取文件并提取数据
file_path_1 = 'smallGrid_DRQN_部分观测r=1_hc不置0.txt'  # 替换为你的txt文件路径
file_path_2 = 'mediumClassic_SNN_energy_realValue.txt'  # 替换为你的txt文件路径
exp_name = "smallGrid_ANN_partial_obs_r=1"

def rl_plotter(file_path_1):
    episode_numbers = []
    mean_rewards = []
    scores = []
    with open(file_path_1, 'r') as file: 
        for line in file:
            # print(line)....................................................
            # 使用正则表达式提取数据
            episode_match = re.search(r"Episode no = (\d+)", line)
            mean_reward_match = re.search(r"mean_reward/100_episodes = (-?\d+\.\d+)", line)
            # score_match = re.search(r"score = (-?\d+\.\d+)", line)
            score_match = re.search(r"mean_score/100_episodes = (-?\d+\.\d+)", line)

            if episode_match and mean_reward_match and score_match:
                episode_numbers.append(int(episode_match.group(1)))
                mean_rewards.append(float(mean_reward_match.group(1)))
                scores.append(float(score_match.group(1)))
        logger = Logger(exp_name = exp_name, env_name='Pacman')
        for i in range(len(episode_numbers)):
            logger.update(score = [mean_rewards[i]],total_steps = i)

def draw_picture_1(file_path_1):
    episode_numbers = []
    mean_rewards = []
    scores = []
    with open(file_path_1, 'r') as file: 
        for line in file:
            # print(line)
            # 使用正则表达式提取数据
            episode_match = re.search(r"Episode no = (\d+)", line)
            mean_reward_match = re.search(r"mean_reward/100_episodes = (-?\d+\.\d+)", line)
            # score_match = re.search(r"score = (-?\d+\.\d+)", line)
            score_match = re.search(r"mean_score/100_episodes = (-?\d+\.\d+)", line)
            if episode_match and mean_reward_match and score_match:
                episode_numbers.append(int(episode_match.group(1)))
                mean_rewards.append(float(mean_reward_match.group(1)))
                scores.append(float(score_match.group(1)))
    # 绘制 mean_reward/100_episodes
    # plt.figure(figsize=(10, 6))
    x = range(len(episode_numbers))
    plt.plot(x, mean_rewards, label='Mean Reward')
    plt.text(x[-1],mean_rewards[-1],' %f' % mean_rewards[-1], ha='left', va='center', fontsize=10)
    plt.title('mean_reward / 100 episodes', fontsize=16)
    plt.xlabel('Index')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图像到文件
    # plt.savefig("result_pic/snn_nolinear_medium_1", dpi=300, bbox_inches='tight')  # dpi 设置图像分辨率
    print(f"图像已保存到save_pic")
    # 显示图表
    # plt.show()

    # 绘制 mean_reward/100_episodes
    # plt.figure(figsize=(10, 6))
    # x = range(len(episode_numbers))
    # plt.plot(x, scores, label='Mean_score')
    # plt.text(x[-1],scores[-1],' %f' % scores[-1], ha='left', va='center', fontsize=10)
    # plt.title('mean_score / 100 episodes', fontsize=16)
    # plt.xlabel('Index')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # 显示图表
    plt.show()


def draw_picture_2(file_path_1,file_path_2):
    episode_numbers = []
    mean_rewards_1 = []
    mean_rewards_2 = []
    mean_reward_match = []
    with open(file_path_1, 'r') as file:
        for line in file:
            # print(line)
            # 使用正则表达式提取数据
            episode_match = re.search(r"Episode no = (\d+)", line)
            mean_reward_match = re.search(r"ann_total_energy = (-?\d+\.\d+)", line)
            if episode_match and mean_reward_match:
                episode_numbers.append(int(episode_match.group(1)))
                mean_rewards_1.append(float(mean_reward_match.group(1)))

    # 绘制 mean_reward/100_episodes
    # plt.figure(figsize=(10, 6))
    x = range(len(episode_numbers))
    plt.plot(x, mean_rewards_1, label='ANN_energy_comsumption')
    plt.text(x[-1],mean_rewards_1[-1],' %f' % mean_rewards_1[-1], ha='left', va='center', fontsize=10)
    plt.title('Energy_comsumption/ 100 episodes', fontsize=16)
    plt.xlabel('Index')
    plt.ylabel('Energy_comsumption')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    episode_numbers = []
    mean_reward_match = []

    with open(file_path_2, 'r') as file:
        for line in file:
            # print(line)
            # 使用正则表达式提取数据
            episode_match = re.search(r"Episode no = (\d+)", line)
            mean_reward_match = re.search(r"snn_total_energy = (-?\d+\.\d+)", line)

            if episode_match and mean_reward_match:
                episode_numbers.append(int(episode_match.group(1)))
                mean_rewards_2.append(float(mean_reward_match.group(1)))

    # plt.figure(figsize=(10, 6))
    x = range(len(episode_numbers))
    plt.plot(x, mean_rewards_2, label='SNN_energy_comsumption')
    plt.text(x[-1],mean_rewards_2[-1],' %f' % mean_rewards_2[-1], ha='left', va='center', fontsize=10)
    plt.title('Energy_comsumption/ 100 episodes', fontsize=16)
    plt.xlabel('Index')
    plt.ylabel('Energy_comsumption')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图像到文件
    # plt.savefig("result_pic/ann_medium_2", dpi=300, bbox_inches='tight')  # dpi 设置图像分辨率
    # 显示图表
    plt.show()

draw_picture_1(file_path_1)
# rl_plotter(file_path_1)
# draw_picture_2(file_path_1,file_path_2)





































# import numpy as np
# import matplotlib.pyplot as plt
# # 设置支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 'Microsoft YaHei'，根据你系统中的字体
# plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示乱码
# # 定义单位正方形的四个顶点
# square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

# # 定义下三角矩阵 L 和 上三角矩阵 U
# L = np.array([[1, 0], [0.5, 1]])  # 你可以调整这些值来看到不同的效果
# U = np.array([[2, 1], [0, 1]])

# # 函数：应用矩阵变换
# def apply_transformation(matrix, points):
#     return np.dot(points, matrix.T)

# # 创建一个图形
# plt.figure(figsize=(8, 8))

# # 原始单位正方形
# plt.plot(square[:, 0], square[:, 1], 'bo-', label="Origin Square", markersize=6)

# # 经过 L 变换后的正方形
# L_transformed = apply_transformation(L, square)
# plt.plot(L_transformed[:, 0], L_transformed[:, 1], 'go-', label="L Transformed Shape", markersize=6)

# # 经过 L 和 U 变换后的最终形状
# LU_transformed = apply_transformation(U, L_transformed)
# plt.plot(LU_transformed[:, 0], LU_transformed[:, 1], 'ro-', label="LU Transformed Shape", markersize=6)

# # 设置图形的标题、坐标轴范围、比例
# # plt.title("LU分解的几何变换")
# plt.xlim(-1, 5)
# plt.ylim(-1, 3)
# plt.gca().set_aspect('equal', adjustable='box')

# # 添加网格
# plt.grid(True)

# # 添加图例
# plt.legend()

# # 显示图形
# plt.show()
