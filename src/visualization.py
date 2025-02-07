
import os
import json
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

plt.rcParams["font.sans-serif"] = "DejaVu Sans Mono"
plt.rcParams['axes.unicode_minus'] = False

# for mini version
# plt.rcParams.update({"font.size":20})  # controls default text size
# plt. rc ('axes', titlesize=20) # fontsize of the title
# plt. rc ('axes', labelsize=20) # fontsize of the x and y labels
# plt. rc ('xtick', labelsize=20) # fontsize of the x tick labels
# plt. rc ('ytick', labelsize=20) # fontsize of the y tick labels
# plt. rc ('legend', fontsize=20) # fontsize of the legend

def create_multi_bars(ax,
                      xlabels,
                      title,
                      datas,
                      errors,
                      colors,
                      groups,
                      tick_step=1,
                      group_gap=0.2,
                      bar_gap=0):
    '''
    生成多组数据的柱状图， refer to https://blog.csdn.net/mighty13/article/details/113873617

    :param ax: 子图对象
    :param xlabels: x轴坐标标签序列
    :param title: 图表标题
    :param datas: 数据集，二维列表，要求列表中每个一维列表的长度必须与xlabels的长度一致
    :param errors: 数据集的误差，二维列表，要求列表中每个一维列表的长度必须与labels的长度一致
    :param colors: 柱子颜色,对应每个组
    :param groups: 每个组的标签。
    :param tick_step: x轴刻度步长，默认为1，通过tick_step可调整x轴刻度步长。
    :param group_gap: 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    :param bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''

    # x为每组柱子x轴的基准位置
    x = np.arange(len(xlabels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap

    error_params = dict(elinewidth=1, capsize=3)  # 设置误差标记参数
    # 绘制柱子
    for i, (y, portion, std, color) in enumerate(zip(datas, groups, errors, colors)):
        ax.bar(x + i * bar_span, y, bar_width, color=color,
               yerr=std, label=f'{portion}', error_kw=error_params)
    ax.set_ylabel('micro-f1 (%)')
    ax.set_title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels)
    ax.legend(title='accuracy', loc='upper right',
              frameon=True, fancybox=True, framealpha=0.7)
    ax.grid(True, linestyle=':', alpha=0.6)

class vis_theis:
    def __init__(self):
        self.base_path = '../data/vis/'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def vis_label_accuracy(self,):
        """

        :param data:
        :param error:
        :param file_name: the file name to be saved
        :return:
        """
        def plot_bars(data, error, file_name, file_type='pdf'):
            models = ['Qwen', 'Mixtral']
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="compressed")
            axs = axs.flatten()

            groups = [1, 0.75, 0.5, 0.25, 0]
            colors = ['#e281b1', '#e89fa7', '#ecb6a1', '#f3cf9c', '#fef795']
            create_multi_bars(axs[0],
                              models,
                              '1-shot',
                              # (2, 5) -> (5, 2), 5个portion，2个模型
                              np.array(data[:2]).T,
                              np.array(error[:2]).T,
                              colors=colors,
                              groups=groups,
                              )
            create_multi_bars(axs[1],
                              models,
                              '5-shot',
                              np.array(data[2:]).T,
                              np.array(error[2:]).T,
                              colors=colors,
                              groups=groups,
                              )
            vis_path = os.path.join(self.base_path, 'part1/label_portion')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            file = os.path.join(vis_path, f'{file_name}.{file_type}')
            print('save file:', file)
            plt.savefig(file, dpi=300)

        # run the model with each dataset and seed

        # 1. conll03
        conll03_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [44.79, 44.55, 45.39, 43.32, 42.23],
            [26.22, 27.32, 27.73, 26.43, 32.26],  # mixtral
            # 5-shot
            [48.67, 47.89, 46.84, 44.16, 42.59],  # qwen
            [43.04, 39.13, 39.45, 40.45, 33.16],  # mixtral
        ]
        conll03_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [1.85, 1.47, 2.05, 3.52, 3.41],
            [1.91, 2.10, 3.54, 3.23, 3.58],  # mixtral
            # 5-shot
            [1.40, 2.85, 1.83, 1.37, 3.79],  # qwen
            [3.41, 4.01, 5.36, 1.37, 1.69],  # mixtral
        ]
        plot_bars(conll03_datas, conll03_errors, 'conll03')

        # 2. ontonotes5
        onto5_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [34.74, 32.50, 28.67, 26.40, 24.13],
            [26.39, 27.57, 25.32, 23.10, 19.59],  # mixtral
            # 5-shot
            [39.49, 31.96, 28.54, 27.33, 21.07],  # qwen
            [16.52, 17.90, 17.31, 13.68, 9.99],  # mixtral
        ]
        onto5_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [1.34, 2.45, 1.65, 1.70, 1.93],
            [2.95, 3.91, 2.30, 2.05, 1.58],  # mixtral
            # 5-shot
            [2.56, 2.77, 3.61, 3.74, 3.09],  # qwen
            [1.87, 1.30, 4.61, 1.50, 2.70],  # mixtral
        ]

        plot_bars(onto5_datas, onto5_errors, 'ontonotes5')

        # 3. movies
        movies_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [67.24, 66.57, 63.81, 59.88, 56.94],
            [68.25, 67.99, 64.75, 60.75, 58.47],  # mixtral
            # 5-shot
            [64.00, 60.65, 54.22, 53.83, 48.38],  # qwen
            [71.02, 70.45, 67.91, 63.22, 50.91],  # mixtral
        ]
        movies_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [0.92, 0.88, 2.80, 1.23, 2.22],
            [1.47, 2.49, 0.95, 2.57, 1.54],  # mixtral
            # 5-shot
            [2.61, 1.58, 2.39, 0.77, 2.57],  # qwen
            [1.04, 2.04, 0.95, 1.78, 4.40],  # mixtral
        ]

        plot_bars(movies_datas, movies_errors, 'movies')

        # 4. restaurant
        restaurant_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [50.27, 50.15, 49.24, 47.18, 43.36],
            [48.71, 49.29, 46.74, 46.81, 43.59],  # mixtral
            # 5-shot
            [62.92, 61.03, 55.97, 53.62, 44.39],  # qwen
            [62.64, 60.82, 56.49, 54.41, 48.12],  # mixtral
        ]
        restaurant_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [2.70, 1.73, 1.49, 0.53, 0.59],
            [1.42, 0.92, 1.84, 1.37, 2.08],  # mixtral
            # 5-shot
            [2.35, 1.15, 1.02, 2.76, 3.16],  # qwen
            [1.17, 2.28, 2.04, 2.60, 2.87],  # mixtral
        ]

        plot_bars(restaurant_datas, restaurant_errors, 'restaurant')


    def vis_subset_size(self, error_bar=False):
        """
        :return:
        """

        datasets = ('conll2003', 'mit_movie', 'ace2005')
        subset_sizes = (0.1, 0.2, 0.3, 0.4, 0.5)
        k_shots = (50, 20, 10, 5)
        seeds = (22, 32, 42)
        augmentation = 'lsp'
        negative_portion = 1
        partition_time = 1
        metric = 'f1'

        colors = ['#F09BA0', '#9BBBE1', '#c0d2ad', '#ffbb8f']  # color for each shot
        error_colors = ['#8D0405', '#060270', '#016c4f', '#f77511']  # error bar color for each shot

        sub_size_path = os.path.join(self.base_path, 'subset_size')
        if not os.path.exists(sub_size_path):
            os.makedirs(sub_size_path)

        for dataset in datasets:
            save_file = os.path.join(sub_size_path, f'{dataset}.pdf')
            fig, ax = plt.subplots(figsize=(5, 5), layout="compressed")
            ax.set_title(dataset)  # 设置标题
            ax.set_xlabel('subset size')  # 设置x轴标签
            ax.set_ylabel('micro-f1 (%)')  # 设置y轴标签
            for shot_idx, k_shot in enumerate(k_shots):
                shot_f1_scores = []  # 每个shot对应一个list
                shot_f1_stds = []

                for subset_size in subset_sizes:
                    aug_method = augmentation
                    aug_method += f'_size-{subset_size}'
                    # if use negative sampling, add post fix '_neg' to the name of augmentation method
                    if negative_portion != 0:  # negative_postfix
                        aug_method += f'_neg-{negative_portion}'
                    aug_method += f'_p-{partition_time}'
                    f1_scores_subset_size = []
                    for seed in seeds:
                        output_dir = '../ckpt/{aug_method}/{dataset}_{k_shot}-shot_{seed}'.format(
                            aug_method=aug_method,
                            k_shot=k_shot,
                            dataset=dataset,
                            seed=seed
                        )  # ckpt and results path
                        result_file = os.path.join(output_dir, 'all_results.json')
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        f1_key = f'eval_{metric}'
                        f1_scores_subset_size.append(results[f1_key])
                    shot_f1_scores.append(np.mean(f1_scores_subset_size))
                    shot_f1_stds.append(np.std(f1_scores_subset_size, ddof=1))  # ddof=1, sample std. ddof=0, population std
                # plot

                if error_bar:
                    ax.errorbar(subset_sizes,
                                shot_f1_scores,
                                yerr=shot_f1_stds,
                                color=colors[shot_idx],
                                fmt='-',
                                marker='o',
                                label=f'{k_shot}-shot',
                                ecolor=error_colors[shot_idx],
                                elinewidth=1,
                                capsize=3)
                else:
                    ax.plot(subset_sizes,
                            shot_f1_scores,
                            '-o',
                            color=colors[shot_idx],
                            label=f'{k_shot}-shot')
                ax.legend(title='k-shot', loc='upper right', frameon=True, fancybox=True)
                # grid
                ax.grid(True, linestyle=':', alpha=0.6)
            print('save file:', save_file)
            fig.savefig(save_file, dpi=300)


if __name__ == '__main__':
    vt = vis_theis()
    vt.vis_subset_size()
