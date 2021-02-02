## RNN中为什么会出现梯度消失？
在RNN中,常用的激活函数为tanh或者sigmoid激活函数, 观察这两个激活函数的导数后,发现都是呈现一种凸二次函数, 且在无穷小和无穷大的时候接近0;  
所以经过神经网络多次梯度传导, 导致梯度越来越小,直至接近0, 发生梯度消失;   



## 如何解决RNN中的梯度消失问题？
- 采用更好的激活函数  
- 设计其他的网络结构, 比如LSTM
- 加入BatchNormalization层,约束网络层每一层的分布   



## LSTM  
RNN中只有单一神经网络层, 例如一个tanh层;  


LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。
LSTM 的关键就是细胞状态，水平线在图上方贯穿运行。细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。

![LSTM4](https://user-images.githubusercontent.com/30361513/81458704-66ab1500-91ce-11ea-88d0-59c90e04669b.png)

LSTM 拥有三个门，分别是忘记层门，输入层门和输出层门，来保护和控制细胞状态。

- 遗忘门
![image](https://user-images.githubusercontent.com/30361513/81458811-15e7ec00-91cf-11ea-9451-b4e056149c83.png)

- 输入门
![image](https://user-images.githubusercontent.com/30361513/81458819-1ed8bd80-91cf-11ea-8e23-5ac651bed72a.png)

- 输出门  
![image](https://user-images.githubusercontent.com/30361513/81458830-2dbf7000-91cf-11ea-9f91-7cf40479ae21.png)

