# 测试线性单元
from linear_unit.LinearUnit import LinearUnit
import matplotlib.pyplot as plt


def get_train_data():
    input_vecs = [[1], [3], [5], [10], [14]]
    labels = [6000, 10000, 15000, 22000, 30000]
    # input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    # 特征数设1个(年限)
    lu = LinearUnit(1)
    input_vecs, labels = get_train_data()
    lu.train(input_vecs, labels, 50, 0.01)
    return lu


# 画图
def plot(linearUnit):
    input_vecs, labels = get_train_data()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(input_vecs, labels)  # 横坐标input_vecs，纵坐标labels
    weights = linearUnit.weights
    bias = linearUnit.bias
    x = range(0, 15, 1)  # 画0到15年的图像
    y = list(map(lambda x: weights[0] * x + bias, x))
    ax.plot(x, y)
    plt.show()


if __name__ == "__main__":
    train_lu = train_linear_unit()
    print(train_lu)
    # 测试
    print('3.4 年,  %.2f' % train_lu.predict([3.4]))
    print('15 年,  %.2f' % train_lu.predict([15]))
    print('1.5 年,  %.2f' % train_lu.predict([1.5]))
    print('6.3 年,  %.2f' % train_lu.predict([6.3]))
    plot(train_lu)
