# Games201HW

---

本人初学者，零基础入门（大二数学基础），因此本教程还算比较舒适，但是也免不了有错误，还请批评指正。

---

# 数值积分

数值积分，是用于求定积分的近似值的一种方法。在数学分析中，有很多计算给定函数的定积分是不可行的，而数值积分是利用黎曼积分等数学定义，用数值逼近的方法近似计算给定的定积分值。借助计算机和编程，数值积分可以快速而有效地计算复杂的积分。

---

# 欧拉方法
欧拉方法是一种数值积分方法，又称为欧拉折线法，是用折线来逼近曲线的一种方法。

例如$\frac{dy}{dx}=f(x,y)$，可以转化为$y_{n+1}-y_n=f(x_n,y_n)h$，其中h则为折线的步长。

由泰勒公式$y(x+h)=y(x)+y^{'}(x)h+o(h)$，可以看出欧拉公式实际上是泰勒公式的离散形式。很显然，h越小，欧拉方法的结果越精确，h越大，结果越不精确。

但是h较大时，除了不精确之外，还会导致使用欧拉方法得到的值不收敛，即不稳定。

欧拉方法分为显式积分和隐式积分两种形式，其中显式积分条件稳定，隐式积分无条件稳定：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706194142548.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)

---
# 弹簧质点系统

在清楚了数值积分的解决方法之后，我们使用它来解决一个最简单的物理模拟问题——弹簧质点系统。

弹簧质点系统中主要有弹力和阻力。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706194956493.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202007061950137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)
总结弹力和阻力的计算公式如下：

$$
\left\{ 
\begin{array}{c}
\LARGE f(x_a)_s=-k_s\frac{x_a-x_b}{||x_a-x_b||}(||x_a-x_b||-l)
\\
\\
\LARGE f(x_a)_d=-k_d\frac{x_a-x_b}{||x_a-x_b||}(x_a^{'}-x_b^{'})·\frac{x_a-x_b}{||x_a-x_b||}
\end{array}
\right. 
$$

## 显式方法
$$
\left\{ 
\begin{array}{c}
\LARGE f_t=\sum_{b=0}^n f(x_a)_s+\sum_{b=0}^nf(x_a)_d
\\
\\
\LARGE v_{t+dt}=v_t+dt*\frac{f_t}{m}
\\
\\
\LARGE x_{t+dt}=x_t+dt*v_t
\end{array}
\right. 
$$

显式方法直接实现即可。

## 隐式方法
$$
\left\{ 
\begin{array}{c}
\LARGE v_{t+dt}=v_t+dt*\frac{f_{t+dt}}{m}
\\
\\
\LARGE x_{t+dt}=x_t+dt*v_{t+dt}
\end{array}
\right. 
$$

隐式方法需要进行特殊的推导。

由泰勒公式的一阶展开，可以得到f的近似：

$\LARGE  f_{t+dt}=f_t+\frac{∂f}{∂x}△x+\frac{∂f}{∂v}△v$

代入上式，可以求得△v的表示形式为：

$\LARGE △v=v_{t+dt}-v_t=dt*\frac{f_{t+dt}}{m}$

$\LARGE △v=\frac{dt}{m}*(f_t+\frac{∂f}{∂x}△x+\frac{∂f}{∂v}△v)$

将△x也表示为△v的式子：

$\LARGE △x=x_{t+dt}-x_t=dt*v_{t+dt}=dt*(v_t+△v)$

代入上式，消去△x：

$\LARGE △v=\frac{dt}{m}*(f_t+dt*\frac{∂f}{∂x}(v_t+△v)+\frac{∂f}{∂v}△v)$

展开括号：

$\LARGE △v=\frac{dt}{m}*f_t+\frac{dt^2}{m}\frac{∂f}{∂x}v_t+\frac{dt^2}{m}\frac{∂f}{∂x}△v+\frac{dt}{m}\frac{∂f}{∂v}△v$

移项：

$\LARGE △v-\frac{dt^2}{m}\frac{∂f}{∂x}△v-\frac{dt}{m}\frac{∂f}{∂v}△v=\frac{dt}{m}*f_t+\frac{dt^2}{m}\frac{∂f}{∂x}v_t$

整理：

$\LARGE (1-\frac{dt^2}{m}\frac{∂f}{∂x}-\frac{dt}{m}\frac{∂f}{∂v})△v=\frac{dt}{m}(f_t+dt*\frac{∂f}{∂x}v_t)$

别忘了我们操作的是一个矩阵：

$\LARGE (I-\frac{dt^2}{M}\frac{∂f}{∂x}-\frac{dt}{M}\frac{∂f}{∂v})△v=\frac{dt}{M}(f_t+dt*\frac{∂f}{∂x}v_t)$

两边乘以M：

$\LARGE (M-dt^2\frac{∂f}{∂x}-dt\frac{∂f}{∂v})△v=dt(f_t+dt*\frac{∂f}{∂x}v_t)$

解隐式积分需要求解上述的线性方程组，并解出△v。

其中$\frac{∂f}{∂x}$和$\frac{∂f}{∂v}$的计算方法在[参考文献](https://blog.mmacklin.com/2012/05/04/implicitsprings/)中写的很清楚，这里直接给出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201005457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201005458.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201005893.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201005826.png)

---

# 雅克比迭代
在隐式方法中，最终需要求解的线性方程组往往是一个巨大的稀疏矩阵，因此很难通过矩阵求逆的方式求解，这里介绍最简单的迭代求解线性方程组的方法——雅克比迭代。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201516551.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706201516503.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)


---

# 实例代码
[弹簧质点系统的显式方法](https://github.com/lilinxi/Games201HW/blob/master/hw1/explicit.py)

[弹簧质点系统的隐式方法](https://github.com/lilinxi/Games201HW/blob/master/hw1/implicit_jacobi_iterative.py)

>代码由python语言以及[taichi框架](https://taichi.readthedocs.io/zh_CN/latest/overview.html)编写而成。

单击鼠标添加质点，相邻质点自动添加弹簧，其中红色弹簧表示伸长中，绿色弹簧表示收缩中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200706195757369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg0MTg0,size_16,color_FFFFFF,t_70)

---

[公式和求导部分参考](https://blog.mmacklin.com/2012/05/04/implicitsprings/)
[隐式方法推导部分参考](https://www.cnblogs.com/shushen/p/5473264.html)
[雅克比迭代部分参考](https://blog.csdn.net/Reborn_Lee/article/details/80959509)