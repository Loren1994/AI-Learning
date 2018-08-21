# 测试线性单元
from linear_unit.LinearUnit import LinearUnit


def get_train_data():
    # input_vecs = [[1], [3], [5], [10], [14]]
    # labels = [6000, 10000, 15000, 22000, 30000]
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    # 特征数设1个(年限)
    lu = LinearUnit(1)
    input_vecs, labels = get_train_data()
    lu.train(input_vecs, labels, 10, 0.1)
    return lu


if __name__ == "__main__":
    train_lu = train_linear_unit()
    print(train_lu)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % train_lu.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % train_lu.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % train_lu.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % train_lu.predict([6.3]))
