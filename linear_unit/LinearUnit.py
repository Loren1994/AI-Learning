# AI - 线性单元
# 可导的线性函数来替代感知器的阶跃函数
# 解决回归问题
from perceptron.Perceptron import Perceptron

# 定义激活函数f
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)
