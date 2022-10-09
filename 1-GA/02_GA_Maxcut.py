import numpy as np
import matplotlib.pyplot as plt

# 交互地 创建邻接矩阵
def Get_Adjecency_Matrix(n, v):
    A = np.zeros((n, n))
    for edge in range(v):
        print("Edge {}: " .format(edge+1), end='' )
        i, j = map(int, input().split())
        A[i-1][j-1] = A[j-1][i-1] = 1
    return A

# 随机初始化种群
def init(pop_size, n):  # 种群数量，染色体长度
    return np.random.randint(0, 2, (pop_size, n))

# 计算适应度
def Fitness(pop, A):
    pop_size = len(pop)
    fit = np.zeros((pop_size, 1))
    for i in range(pop_size):
        s = np.nonzero(np.where(pop[i], 0, 1))
        v_s = np.nonzero(pop[i])
        fit[i] = np.sum(A[s][:, v_s])
    return fit

# 轮盘赌选择
def roulette_wheel( pop, pop_size, fit):
    p = (fit/fit.sum()).cumsum()
    new_pop = []
    for i in range(pop_size):
        select_p = np.random.uniform()
        new_pop.append(pop[p > select_p][0])
    return np.array(new_pop)
# def roulette_wheel(pop_size, pop, fit):
#     p = (fit/fit.sum()).cumsum()
#     yield_num = 0
#     while (yield_num < pop_size):
#         yield_num += 1
#         select_p = np.random.uniform()
#         yield pop[p>select_p][0]

# 基因突变
def mutation(new_pop, pm):
    pop_size, n = np.shape(new_pop)
    for i in range(pop_size):
        if np.random.uniform() < pm:
            # 染色体上的随机突变点
            point = np.random.randint(0, n)
            new_pop[i][point] = 1 - new_pop[i][point]
    return new_pop

# 基因重组
def Pantogamy(new_pop):
    pop_size, n = np.shape(new_pop)
    new = []
    while len(new) < pop_size:
        # if np.random.uniform() < pcross:
        mother_idx, father_idx = np.random.randint(0, pop_size, 2)
        cross_point = np.random.randint(0, n)
        child1 = np.concatenate([
            new_pop[father_idx][:cross_point],
            new_pop[mother_idx][cross_point:]
        ])
        child2 = np.concatenate([
            new_pop[mother_idx][:cross_point],
            new_pop[father_idx][cross_point:]
        ])
        new.extend([child1, child2])  # 依次循环增加2个新个体
    return np.array(new)  # 由<list> 转为 <numpy.ndarray>

if __name__ == "__main__":
    pm = 0.01  # 突变概率
    # pc = 0.90  # 交叉概率
    pop_size = 30  # 种群数量
    Iter = 50  # 迭代上限
    iter_plot = []  # 绘图——横坐标 代数
    fit_max_plot = []  # 绘图——纵坐标 最佳适应度

    n = input("输入节点个数: ")
    n = eval(n)
    v = input("输入边个数: ")
    v = eval(v)
    A = Get_Adjecency_Matrix(n, v)
    population = init(pop_size, n)

    for iTer in range(Iter):
        fit = Fitness(population, A)
        # 这一代的最佳个体追加到绘图向量
        iter_plot.append(iTer)
        fit_max = np.max(fit)
        fit_max_plot.append(fit_max)
        # 轮盘赌得到子代
        new_pop = roulette_wheel(population, pop_size, fit)
        # 基因突变
        new_pop_after_mutation = mutation(new_pop, pm)
        # 基因重组 (必须返回population，使代代连接)
        population = Pantogamy(new_pop_after_mutation)
    fit_final = Fitness(population, A)
    # 找到fit_final最大值对应的下标
    max_fit_idx = fit_final.argmax()
    # 最大适应度
    max_fit = fit_final[max_fit_idx]
    # 最优解
    max_cut = population[max_fit_idx]
    # 最大适应度追加到 展示数组中
    fit_max_plot.append(max_fit)
    iter_plot.append(Iter)

    print("近似最优解: ", max_cut)
    print("近似最优适应度: %d" % max_fit)
    # 画图
    plt.figure()
    plt.plot(iter_plot, fit_max_plot)
    # plt.title("Max-Cut problem solved by GA , Shan Yiwen, SWU")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel(r"$\vert \delta(S) \vert$", fontsize=20)
    plt.show()
