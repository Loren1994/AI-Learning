# AI - 感知器
# 激活函数为阶跃函数 - 解决分类问题

# 3和2差异
# lambda a, b: a + b, map(lambda (x, w): x * w, zip(input_vec, self.weights))
# 应改为 [x * w for x, w in zip(input_vec, self.weights)] // lambda (x,w):x*w => lambda x:x[0]*x[1]
# 打印map => print(list(map))
# reduce内置函数需导入from functools import reduce

from functools import reduce


class Perceptron():
    # 初始化感知器,设置参数个数,激活函数
    def __init__(self, input_num, activator):
        # 激活函数
        self.activator = activator
        # 权重
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项
        self.bias = 0.0

    # 学习到的权重、偏置项
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (list(self.weights), self.bias)

    # 输入向量,输出感知器结果
    # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
    # 变成[(x1,w1),(x2,w2),(x3,w3),...]
    # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
    # 最后利用reduce求和
    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a, b: a + b,
                   [(x * w) for (x, w) in zip(input_vec, self.weights)], 0.0) + self.bias)

    # 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_train(input_vecs, labels, rate)

    # 一次训练,把所有的训练数据过一遍
    def _one_train(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weight(output, input_vec, label, rate)

    def _update_weight(self, output, input_vec, label, rate):
        delta = label - output
        self.weights = [(w + rate * delta * x) for (x, w) in zip(input_vec, self.weights)]
        self.bias += rate * delta
