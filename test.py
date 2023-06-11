# 导入库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成数据集
x = np.linspace(0, 10, 100) # 生成100个在0到10之间均匀分布的数作为x
y = 3 * x + 5 + np.random.normal(0, 1, 100) # 生成y = 3x + 5 + 噪声

# 划分训练集和测试集
x_train = x[:80] # 前80个作为训练集
y_train = y[:80]
x_test = x[80:] # 后20个作为测试集
y_test = y[80:]

# 定义模型
model = tf.keras.Sequential() # 创建一个序贯模型
model.add(tf.keras.layers.Dense(1, input_shape=(1,))) # 添加一个全连接层，输出维度为1，输入维度为1

# 定义损失函数和优化器
model.compile(loss='mse', optimizer='sgd') # 使用均方误差作为损失函数，随机梯度下降作为优化器

# 训练模型
history = model.fit(x_train, y_train, epochs=50) # 训练50个周期

# 评估模型
model.evaluate(x_test, y_test) # 计算测试集上的损失

# 可视化模型
plt.scatter(x, y) # 绘制数据点
plt.plot(x, model.predict(x), 'r') # 绘制拟合曲线
plt.show()
