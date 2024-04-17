import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable

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

dp = {}
dp.setdefault('a', [[0, 0, 0, 0, 0, 0, 0],
                    [1 / 5, 0, 0, 0, 0, 0, 0],
                    [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                    [44 / 45, - 56 / 15, 32 / 9, 0, 0, 0, 0],
                    [19372 / 6561, - 25360 / 2187, 64448 / 6561, - 212 / 729, 0, 0, 0],
                    [9017 / 3168, - 355 / 33, 46732 / 5247, 49 / 176, - 5103 / 18656, 0, 0],
                    [35 / 384, 0, 500 / 1113, 125 / 192, - 2187 / 6784, 11 / 84, 0]])

dp.setdefault('b', [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
dp.setdefault('b_star', [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
dp.setdefault('rank', [4, 5])

dp8 = {}
dp8.setdefault('a', [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1 / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1 / 48, 1 / 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1 / 32, 0, 3 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5 / 16, 0, -75 / 64, 75 / 72, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3 / 80, 0, 0, 3 / 16, 3 / 20, 0, 0, 0, 0, 0, 0, 0, 0],
                     [29443841 / 614563906, 0, 0, 77736538 / 692538347, -28693883 / 1125000000, 23124283 / 1800000000,
                      0, 0, 0, 0, 0, 0, 0],
                     [16016141 / 946692911, 0, 0, 61564180 / 158732637, 22789713 / 633445777, 545815736 / 2771057229,
                      -180193667 / 1043307555, 0, 0, 0, 0, 0, 0],
                     [39632708 / 573591083, 0, 0, -433636366 / 683701615, -421739975 / 2616292301,
                      100302831 / 723423059, 790204164 / 839813087, 800635310 / 3783071287, 0, 0, 0, 0, 0],
                     [246121993 / 1340847787, 0, 0, -37695042795 / 15268766246, -309121744 / 1061227803,
                      -12992083 / 490766935, 6005943493 / 2108947869, 393006217 / 1396673457, 123872331 / 1001029789, 0,
                      0, 0, 0],
                     [-1028468189 / 846180014, 0, 0, 8478235783 / 508512852, 1311729495 / 1432422823,
                      -10304129995 / 1701304382, -48777925059 / 3047939560, 15336726248 / 1032824649,
                      -45442868181 / 3398467696, 3065993473 / 597172653, 0, 0, 0],
                     [185892177 / 718116043, 0, 0, -3185094517 / 667107341, -477755414 / 1098053517,
                      -703635378 / 230739211, 5731566787 / 1027545527, 5232866602 / 850066563, -4093664535 / 808688257,
                      3962137247 / 1805957418, 65686358 / 487910083, 0, 0],
                     [403863854 / 491063109, 0, 0, -5068492393 / 434740067, -411421997 / 543043805,
                      652783627 / 914296604, 11173962825 / 925320556, -13158990841 / 6184727034,
                      3936647629 / 1978049680, -160528059 / 685178525, 248638103 / 1413531060, 0, 0]])
dp8.setdefault('b', [13451932 / 455176623, 0, 0, 0, 0, -808719846 / 976000145, 1757004468 / 5645159321,
                     656045339 / 265891186, -3867574721 / 1518517206, 465885868 / 322736535, 53011238 / 667516719,
                     2 / 45, 0])


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


def rk_adaptive(t0, t_end, y0, h_init, f, tableau, Atoli, Rtoli):
    t_limit = int((t_end - t0) / h_init)
    t = np.zeros(t_limit)
    y = np.zeros((y0.size, t_limit))
    y_star = np.zeros((y0.size, t_limit))

    y[:, 0] = y0
    h = h_init
    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1], y_star[:, step + 1], h = rk_one_step_adaptive(float(t[step]), y[:, step],
                                                                                   y_star[:, step], h, f, tableau,
                                                                                   Atoli, Rtoli)

    return t, y


def rk_one_step_adaptive(t, y, y_star: np.ndarray, h, f, tableau, Atoli, Rtoli):
    k = k_count(y, h, f, tableau)
    y_n = y
    y_n_star = y_star
    b_ = tableau['b']
    b_star = tableau['b_star']
    p, p_star = tableau['rank']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
        y_n_star += h * b_star[i] * k[i]
    t_n = t + h

    local_error = h * np.sum([(b_[i] - tableau['b_star'][i]) * k[i] for i in range(len(b_))])
    tol_first = Atoli + np.maximum(np.fabs(y_n[0]), np.fabs(y[0])) * Rtoli
    tol_second = Atoli + np.maximum(np.fabs(y_n[1]), np.fabs(y[1])) * Rtoli
    err = np.sqrt((1 / 2) * ((local_error / tol_first) ** 2 + (local_error / tol_second) ** 2))
    h_new = h * ((1.0 / err) ** (1.0 / (min(p, p_star) + 1)))

    return t_n, y_n, y_n_star, h_new


def particle(arr):
    # x = arr[:3]
    v = arr[3:]

    m = 1.6e-29
    q = 9.1e-31
    e = np.array([0., 1., 0.])
    b = np.array([1., 0., 0.])
    new = np.concatenate((v, (q / m) * (e + np.cross(v, b))), axis=None)
    vec = np.array(new)

    return vec


def rel_particle(arr):
    # x = arr[:3]
    v = arr[3:]
    m = 1.6e-29
    q = 9.1e-31
    c = 3e8
    e = np.array([0., 1., 0.])
    b = np.array([1., 0., 0.])

    new = np.concatenate((v, (q / m) * ((e + np.cross(v, b)) /
                                        (((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) *
                                          (1 / (1 - ((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) / c ** 2))) ** (
                                                  3 / 2)) / c ** 2 +
                                         1 / (np.sqrt(1 - (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) / c ** 2))))), axis=None)
    vec = np.array(new)
    return vec


def main():
    # НАЗВАНИЕ ТАБЛИЦЫ
    table = rk4

    # НАЧАЛЬНЫЕ ДАННЫЕ
    y0 = np.array([0., 0., 0., 1e8, 1e8, 100.])
    t, y1 = runge_kutta(0, 1000, y0, 1 / 100, particle, table)
    t, y2 = runge_kutta(0, 1000, y0, 1 / 100, rel_particle, table)

    t, etalon = runge_kutta(0, 1000, y0, 1 / 100, particle, dp8)
    # t, y1 = rk_adaptive(0, 100, y0, 0.01, rel_particle, table, Atoli=1e-7, Rtoli=1e-6)
    # t, y2 = rk_adaptive(0, 100, y0, 0.01, rel_particle, table, Atoli=1e-7, Rtoli=1e-6)

    error1 = 0
    error2 = 0
    for i in range(etalon[0].size - 1):
        x1 = np.sqrt((y1[0][i] ** 2 + y1[1][i] ** 2 + y1[2][i] ** 2))
        x2 = np.sqrt((y2[0][i] ** 2 + y2[1][i] ** 2 + y2[2][i] ** 2))
        etalon_x = np.sqrt((etalon[0][i] ** 2 + etalon[1][i] ** 2 + etalon[2][i] ** 2))

        error1 += ((x1 - etalon_x) ** 2) / (max(x1, etalon_x) ** 2)
        error2 += ((x2 - etalon_x) ** 2) / (max(x2, etalon_x) ** 2)
    error = np.sqrt((1 / etalon[0].size) * max(error1, error2))

    v_x_1 = y1[3]
    v_y_1 = y1[4]
    v_z_1 = y1[5]
    v_1 = np.sqrt(v_y_1 ** 2 + v_x_1 ** 2 + v_z_1 ** 2)
    v_x_2 = y2[3]
    v_y_2 = y2[4]
    v_z_2 = y2[5]
    v_2 = np.sqrt(v_y_2 ** 2 + v_x_2 ** 2 + v_z_2 ** 2)
    delta = np.fabs(v_1 - v_2)
    for i in range(len(delta)):
        if delta[i] >= error:
            print(v_1[i])
            break
    print(error)
    # Добавление трехмерного графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение графика
    xs = y1[0]
    ys = y1[1]
    zs = y1[2]
    ax.scatter(xs, ys, zs)

    # Отображение графика
    plt.show()


if __name__ == "__main__":
    main()
