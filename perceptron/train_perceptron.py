# 感知器训练测试

from perceptron.Perceptron import Perceptron


# 定义激活函数
def f(x: int):
    return 1 if x > 0 else 0


# 基于and真值表构建训练数据
def get_training_data_sets():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


# 使用and真值表训练感知器
def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_data_sets()
    p.train(input_vecs, labels, 10, 0.1)
    return p  # 训练好的感知器


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perceptron)
    # 测试
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))
