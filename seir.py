import numpy as np
import matplotlib.pyplot as plt

# BUTCHER TABLES
euler = {}
euler.setdefault('a', [[0]])
euler.setdefault('b', [1])

midpoint = {}
midpoint.setdefault('a', [[0, 0],
                          [1 / 2, 0]])
midpoint.setdefault('b', [0, 1])

rk4 = {}
rk4.setdefault('a', [[0, 0, 0, 0],
                     [1 / 2, 0, 0, 0],
                     [0, 1 / 2, 0, 0],
                     [0, 0, 1, 0]])
rk4.setdefault('b', [1 / 6, 1 / 3, 1 / 3, 1 / 6])


def k_count(y, h, f, table):
    a_ = table['a']
    k = []

    y_n = np.copy(y)
    for i in range(len(a_)):
        for j in range(i):
            y_n += h * a_[i][j] * k[j]
        k.append(f(y_n))

    return k


def one_stage(t, y, h, f, table):
    k = k_count(y, h, f, table)
    y_n = y
    b_ = table['b']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h

    return t_n, y_n


def runge_kutta(t_start, t_finish, y0, h, f, table):
    steps_num = int((t_finish - t_start) / h)
    t = np.zeros(steps_num)

    y = np.zeros((y0.size, steps_num))
    y[:, 0] = y0

    for step in range(steps_num - 1):
        t[step + 1], y[:, step + 1] = one_stage(t[step], y[:, step], h, f, table)

    return t, y


def seir(arr):
    """
    :param arr:  [S, E, I, R]
    """
    N = sum(arr)
    S = arr[0]
    E = arr[1]
    I = arr[2]
    R = arr[3]

    alpha = 1 / 10  # коэффициент инкубации, число обратное продолжительности инкубационного периода
    beta = 8 / 10  # вероятность заразиться при контакте инфицированного с восприимчивым
    gamma = 1 / 10  # коэффициент восстановления, обратный продолжительности болезни
    mu = 1 / 100  # рождаемость - смертность (предполагается, что они равны для поддержания постоянной численности населения)

    # без динамики рождаемости и смертности
    # vec = np.array([- ((beta * I * S) / N),
    #                 ((beta * I * S) / N) - alpha * E,
    #                 alpha * E - gamma * I,
    #                 gamma * I])

    # с динамикой рождаемости и смертности
    vec = np.array([mu * N - mu * S - ((beta * I * S) / N),
                    ((beta * I * S) / N) - (mu + alpha) * E,
                    alpha * E - (gamma + mu) * I,
                    gamma * I - mu * R])

    return vec


def main():
    # СЮДА ВБИВАЕШЬ НАЗВАНИЕ ТАБЛИЦЫ
    table = rk4

    # НАЧАЛЬНЫЕ ДАННЫЕ
    y0 = np.array([99, 10, 0, 0])
    t, y = runge_kutta(0, 100, y0, 1 / 100, seir, table)

    plt.figure(figsize=(10, 6))
    colors = ['r', 'b', 'y', 'g']
    arr = ['S', 'E', 'I', 'R']

    for i in range(len(y)):
        plt.plot(t, y[i, :], colors[i], label=arr[i])
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
