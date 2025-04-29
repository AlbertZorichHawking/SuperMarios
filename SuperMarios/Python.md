### Python

**1.python不用事先声明变量，赋值过程中就包含了变量声明和定义的过程** 

**2.用“=”赋值，左边是变量名，右边是变量的值**
#### 变量
##### 一、数字
1.整数
int_var = 1
2.浮点数
float_var = 1.0
3.复数
这个不讲了，用的不多。

##### 二、字符串
```python
str = 'Hello World!'
print (str) # 输出完整字符串
print (str[0]) # 输出字符串中的第一个字符
print (str[2:5]) # 输出字符串中第三个至第五个之间的字符串
print (str[2:]) # 输出从第三个字符开始的字符串
print (str[::-1])#从倒数第一个字符开始输出
print (str * 2) # 输出字符串两次
print str + "TEST" # 输出连接的字符串

```
运行以上代码的结果：
```
Hello World!
H
llo
llo World!
！dlorW olleH
Hello World!Hello World!
Hello World!TEST
```
##### 三、列表
类似于C++或Java语言的数组，一个有序可变集合的容器。支持内置的基础数据结构甚至是列表，列表是可以嵌套的。不同的数据结构也可以放在同一个列表中，没有统一类型的限制。
```python
list_a = ["str", 1, ["a", "b", "c"], 4]
list_b = ["hello"]
print (list_a[0])
print (list_a[1:3])
print (list_a[1:])
print (list_b * 2)
print (list_a + list_b)
```
运行以上代码：
```
str
[1, ['a', 'b', 'c']]
[1, ['a', 'b', 'c'], 4]
['hello', 'hello']
['str', 1, ['a', 'b', 'c'], 4, 'hello']
```
##### 四、元组
可以视为不可变的列表，在赋值之后不可二次更改。
```python
tuple_a = ("str", 1, ["a", "b", "c"], 4)
tuple_b = ("hello",)
print (tuple_a[0])
print (tuple_a[1:3])
print (tuple_a[1:])
print (tuple_b * 2)
print (tuple_a + tuple_b)
```
##### 五、字典
类似于C++语言的map，key-value键值对的集合，无序的容器。
```python
dict_a = {
    "name": "Alan",
    "age": 24,
    1: "level_1"
}
print (dict_a["name"])
print (dict_a["age"])
print (dict_a[1])
print ("name" in dict_a)
print ("xxx" in dict_a)
print (dict_a.keys())
print (dict_a.values())
print (dict_a.items())
```
运行以上代码：
```
Alan
24
level_1
True
False
[1, 'age', 'name']
['level_1', 24, 'Alan']
[(1, 'level_1'), ('age', 24), ('name', 'Alan')]
```
### python运算符
#### 一、算术运算符
```
+       加
-       减
*       乘
/        除
%        取模
**       幂
//       整除
```
#### 二、比较运算符
```
==       等于
!=        不等于
>         大于
<         小于
>=      大于 或等于
<=      小于或等于
```

#### 三、赋值运算符
```
=        赋值
+=       a += b 等价于 a = a + b
-=          a -= b 等价于 a = a - b
*=          a *= b 等价于 a = a * b
/=          a /= b 等价于 a = a / b
%=        a %= b 等价于 a = a % b
**=        a **= b 等价于 a = a ** b
//=       a //= b 等价于 a = a // b
```

#### 语句
##### 一、if
**If（else if/elif） 条件:**
    满足条件执行的语句
**else:**
    不满足条件执行的语句
**例如**
```python
a = 2
if a == 1:
    print ("a == 1")
elif a == 2:
    print ("a == 2")
else:
    print ("a != 1 and a != 2")
```
##### 二、for
**用来遍历容器、或者执行重复性的代码。**

1、遍历容器
```python
list_a = [1, 2, "test"]
for i in list_a:
print (i)
```
运行结果
```
1
2
test
```
2、执行重复性代码
```python
for i in range(0, 10):
    print i
```
运行结果
```
0
1
2
3
4
5
6
7
8
9
```
##### 三、While
用来执行重复的代码
##### 四、break
终止当前的循环
##### 五、continue
继续当前的循环
#### python list、tuple、dict、set
##### 一、list
python内置的一种数据结构，有序，可更改（添加、删除）
1、声明
```python
>>> game = ["dota", "dota2", "lol"]
>>> game
['dota', 'dota2', 'lol']
```
2、获取列表长度
```python
>>> len(game)
3
```
3、获取元素
```python
>>> game[0]
'dota'
>>> game[1]
'dota2'
>>> game[2]
'lol'
>>> game[3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> 
如果下标越界会报错
```
4、更改
```python
>>> game[0] = "dota3"
>>> game
['dota3', 'dota2', 'lol']
```
5、增加元素
（1）、末尾追加
```python
>>> game.append("wow")
>>> game[3]
'wow'
>>> game
['dota', 'dota2', 'lol', 'wow']
```
（2）指定位置插入
```python
>>> game.insert(2, "war3")
>>> game
['dota', 'dota2', 'war3', 'lol', 'wow']
```
6、删除元素
（1）、删除末尾的元素
```python
>>> game.pop()
'wow'
>>> game
['dota', 'dota2', 'war3', 'lol']
```
（2）、删除指定位置元素
```python 
>>> game.pop(1)
'dota2'
>>> game
['dota', 'war3', 'lol']
```
##### 二、tuple
**python内置的一种数据结构，有序，不可更改，在赋值的时候决定所有元素**
1、声明
```python
>>> game = ('dota', 'war3', 'lol')
>>> game
('dota', 'war3', 'lol')
```
2、获取长度
```python
>>> len(game)
3
```
3、获取元素
```python
>>> game[0]
'dota'
>>> game[1]
'dota2'
>>> game[2]
'lol'
>>> game[3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> 
```
如果下标越界会报错
##### 三、dict
python内置的一种数据结构，无序，可更改
类似于C++语言的map，存键值对，有很快的查找速度。虽然用list遍历也可以得到结果，但是太慢了。 dict是典型的用空间换时间的例子。会占用大量内存，但是查找、插入速度很快，不会随着元素数量增加而增加。 list则是时间换空间的例子，不会占用大量内存，但是随着元素数量增多，查找时间会变很长。
1、声明
```python
>>>name = {1: "alan", 2: "bob", 3: "lucy"}
```
2、查找
```python
>>>name[1]
'alan'
>>>name[2]
'bob'
>>>name[3]
'lucy'
>>> name[5]
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
KeyError: 5
## 为了避免出现报错的情况我们一般事先判断一下这个key是否在字典中
>>> 5 in name
False
##也可以用get方法取，如果key不存在，会返回None或者自己定义的默认值
>>> name.get(5)
>>> name.get(5, "default")
'default'
```
3、更改
```python
>>> name[4] = "zorich"
>>> name
{1: "alan", 2: "bob", 3: "lucy", 4: "zorich"}
```
4、删除
```python
>>> name.pop(1)
'alan'
>>> name
{2: 'bob', 3: 'lucy'}
```
5、元素个数
```python
>>> len(name)
2
```
6、获取所有key
```python
>>> name.keys()
[2, 3]
```
7、获取所有value
```python
>>> name.values()
['bob', 'lucy']
```
8、获取所有键值对
```python
>>> name.items()
[(2, 'bob'), (3, 'lucy')]
```
##### 四、set
python内置数据结构，无序，可更改
set可以视为没有value的dict，只存key，一般用做去重或者集合求交、求并等。
1、声明
```python
>>> girls_1 = set(['lucy', 'lily'])
>>> girls_2 = set(['lily', 'anna'])
```
2、求并、交
```python
>>> girls_1 & girls_2
set(['lily'])
>>> girls_1 | girls_2
set(['lily', 'lucy', 'anna'])
```
3、增加元素
```python
>>> girls_1.add('marry')
>>> girls_1
set(['marry', 'lily', 'lucy'])
```
4、删除元素
```python
>>> girls_1.remove('lucy')
>>> girls_1
set(['marry', 'lily'])
```
#### 函数
函数是可重复调用的代码段，能提高代码的复用率。
定义格式
```python
#无参数
def print_hello():
    print ("hello")
print_hello()
#带参数
def print_str(s):
    print (s)
    return s * 2
print_str("python")
#带默认参数
def print_default(s="hello"):
    print (s)
print_default()
print_default("default")
#不定长参数
def print_args(s, *arg):
    print (s)
    for a in arg:
        print (a)
    return
print_args("hello")
print_args("hello", "world", "1")
# 参数次序可以变
def print_two(a, b):
    print a,b
print_two(a="a", b="b")
print_two(b="b", a="a")
```
输出结果：
```
hello

python

hello

default

hello

hello

world

1

a b

a b
```
#### 类
简介
面向对象是我们经常能听到的术语，即class，类。事实上，主角是两个，一个是类，一个是类实例。人类，是一个类，我们每一个人是一个人类的实例。而类之间又有一些关系，例如，我们既是人类，也是动物，更细化来讲，我们是哺乳类动物，灵长类，类似于集合的概念，哺乳动物属于动物，而在面向对象中我们通常称哺乳动物是动物的子类。而对于动物这个类来说，会自带一些属性，例如：年龄、体重。也会有一些方法：生殖、呼吸。而不同种类的动物（即动物类的各种子类）可能会有不同的属性或方法，像胎生、卵生，像鸟类的飞行的方法和豹子奔跑的方法。
##### 一、定义
用关键字class去定义一个类，如果没有指定父类，默认继承object类。
```python
class Human(object):
    pass
```
这样，我们定义个了一个Human，人类。
##### 二、类属性
```python
class Human(object):
    taisheng = True
```
为什么要叫类属性呢，因为这个属性是和类绑定的，并不是和实例绑定的。胎生这个属性是全人类共有的，并不是某个人特殊拥有的属性。
##### 三、实例属性
```python
class Human(object):
    def __init__(self, name):
        self.name = name
human_a = Human("alan")
```
我们首先实例化了一个人类human_a，然后给这个人类设置了一个实例属性name，name这个属性独立于其他的人类，是和实例绑定的，所以叫实例属性。
实例属性可以在实例创建后任意时间设置。
一般放在构造函数里__init()__
##### 四、类方法
```python
class Human(object):
    def __init__(self, name):
        self.name = name
    def walk(self):
        print (self.name + " is walking")
human_a = Human("alan")
human_a.walk()
```
运行结果：
```
alan is walking
```
类的方法可以看做是一种类属性，而传入的第一个参数self，表示调用这个类方法的实例。像上面的例子，human_a调用了walk这个类方法，human_a的名字是alan，所以运行的结果就是alan is walking。
##### 五、访问控制
从上面的例子来看，我们可以在外部随意更改name这个属性，如果不想让外部直接访问到，则在属性名字前加下划线__name，这样从外部就无法直接访问了。如果还是想访问，可以再加个get的接口。
```python
class Human(object):
    def __init__(self, name):
        self.__name = name
    def walk(self):
        print (self.name + " is walking") 
    def get_name(self):
        return self.__name
human_a = Human("alan")
print (human_a.get_name())
print (human_a.__name)
#如果还是想更改__name字段，可以再加上一个set接口
class Human(object):
    def __init__(self, name):
        self.__name = name
    def walk(self):
        print (self.name + " is walking")
    def get_name(self):
        return self.__name
    def set_name(self, name):
        self.__name = name
human_a = Human("alan")
print (human_a.set_name("bob"))
```
可能有人会有疑问，为何要这么“画蛇添足”呢？其不然，这样会增强代码的健壮性，直接暴漏属性可能会带来意想不到的后果，通过接口的方式可以加以控制，例如，我们可以通过set接口去限定name的长度。
```python
class Human(object):
    def __init__(self, name):
        self.__name = name
    def walk(self):
        print (self.name + " is walking")
    def get_name(self):
        return self.__name
    def set_name(self, name):
        if len(name) <= 10:
            self.__name = name
human_a = Human("alan")
print (human_a.set_name("bob"))
```
这样就不会出现name过长的情况。
六、继承
最开始的简介里说到，哺乳动物是动物的一种，用面向对象的属于来说，哺乳动物是动物的子类，子类拥有父类的属性、方法，即继承。同时又可以拥有父类没有的属性和方法，即多态。 还是以人类为例，通常来说，人类又可以分为男人和女人（当然也有别的，23333）
```python
class Man(Human):
    def __init__(self, name, has_wife):
        self.__name = name
        self.__has_wife = has_wife
```
我们看下这个男人，多了一个新的属性，__has_wife(是否已婚)。我们写到了Man的构造函数里。其实通常并不这么写构造函数，假如Human里有很多属性、很多初始化步骤，通常会这么写。
```python
class Man(Human):
    def __init__(self, name, has_wife):
        super(Man, self).__init__(name)
        self.__has_wife = has_wife
super(Man, self).__init__(name)
```
等价于调用了父类Human的构造函数，就不用再复制粘贴一遍了。 既然有男人，那就再来个女人
```python
class Woman(Human):
    def __init__(self, name, has_husband):
        super(Woman, self).__init__(name)
        self.__has_husband = has_husband
```
我们都知道，男人和女人是不一样的，通常男人都自带抽烟、喝酒、烫头，啊。。。并没有烫头。

```python
class Man(Human):
    def __init__(self, name, has_husband):
        super(Man, self).__init__(name)
        self.__has_husband = has_husband
    def smoke(self):
        print "A man maybe smoke"
    def drink(self):
        print "A man maybe drink"
```
当然，女人也可能自带逛街、化妆等天赋技能。
```python
class Woman(Human):
    def __init__(self, name, has_husband):
        super(Woman, self).__init__(name)
        self.__has_husband = has_husband
    def shopping(self):
        print ("A woman maybe go shopping")
    def make_up(self):
        print ("A woman maybe make up")
```
#### 模块
通常来说，比较正规的工程不会把所有代码放在一个py文件里，我们会把代码拆成各个模块，分别调用。对python来说，拆成各个模块可以看做拆成各个py文件。
### 一、搜索路径
同文件夹下的py文件可以直接import。
```python
def print_hello():
    print ("hello")
```
我们把这个保存至hello.py
```python
import hello
hello.print_hello()
```
在run.py里import，然后调用print_hello() 目录结构
...../
     hello.py
     run.py
hello.py和run.py在同一目录下，可以直接import 如果在不同路径下，可以在sys.path里手动加入你想import的路径
```python
import sys
sys.path.append('/document/document2/document3')
import hello
hello.print_hello()
```
包
通常一个工程不可能只有一层目录结构，并且也不会一个一个path去append到sys里，常用的做法是包，一个目录及其子目录组成的一个包
例
/documents/code/python
├── __ init__.py
├── __ init__.pyc
├── m1
│   ├── b.py
│   ├── b.pyc
│   ├── __ init__.py
│   ├── __ init__.pyc
│   └── m1_1
│       ├── a.py
│       ├── a.pyc
│       ├── __ init__.py
│       └── __ init__.pyc
└── m2
    ├── __init__.py
    └── run.py
这是一个python文件夹，里面有m1和m2这两个文件夹，同时m1中又有一个子文件夹m1_1。 文件b.py
```python
def hello_b():
    print ("hello b")
```
文件a.py
```python
def hello_a():
    print ("hello a")
```
文件run.py
```python
import sys
import os
sys.path.append('/documents/code/')
from course.m1 import b
from course.m1.m1_1 import a
if __name__ == '__main__':
    b.hello_b()
    a.hello_a()
```
在run.py中要调用m1/b.py和m1/m1_1/a.py，只需要导入course这个包就可以了。
#### 字符串处理
1、查找
```python
>>> s = "abc"
>>> s.find("b")
1
>>> s.find("bc")
1
>>> s.find("xx")
-1
```
查找时，返回的是第一个匹配的子串的下标位置，如果没有找到，返回-1
2、分割
字符串按照某个子串进行分割，返回分割后得到列表
```python
>>> s = "aa12bb12cc"
>>> s.split('12')
['aa', 'bb', 'cc']
```
3、大小写转换
```python
>>> s = "abc"
>> s.upper()
'ABC'
>>> s = "ABC"
>>> s.lower()
'abc'
```
4、截取
```python
>>> s = "1234567"
>>> s[2:5]
'345'
>>> s[:5]
'12345'
>>> s[3:]
'4567'
>>> s[3:-1]
'456'
```
追加
```python
>>> s = "123"
>>> t = "456"
>>> s + t
'123456'
```
5、替换
```python
>>> s = "1,2,3"
>>> s.replace(",", "#")
'1#2#3'
```
6、连接
```python
>>> s = ['a', 'b', 'c']
>>> ",".join(s)
'a,b,c'
```
反转
```python
>>> s = "abc"
>>> s[::-1]
>>> 'cba'
```

#### Packet
##### 一、NumPy
Python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。Numpy内部解除了Python的PIL(全局解释器锁),运算效率极好,是大量机器学习框架的基础库
Numpy提供的新的数据类型

* 8、16、32、64位的整型

* 16、32、64位的浮点数

* 64、128位的复数


**Ndarray结构化数组类型**


**1、多维数组**
https://www.cnblogs.com/xzcfightingup/p/7598293.html
**2、向量**
https://blog.csdn.net/zenghaitao0128/article/details/78300770
**3、多项式函数**
Polyld类允许我们用自然的方式来处理多项式函数，接受降幂的系数数组作为参数
```python
p=np.poly1d([2,3,4])

print (np.poly1d(p))
```
运行结果   

```
2

2 x + 3 x + 4
```
可进行定点求值
```python
In[16]: p(2)
Out[16]: 18
```
求根
```python
In[17]:p.r

Out[17]: array([-0.75+1.19895788j, -0.75-1.19895788j])
```
##### 二、Matplotlib
Matplotlib 是一个 Python 的 2D绘图库，它以各种硬拷贝格式和跨平台的交互式环境生成出版质量级别的图形

主要用到pyplot子包

```python
import numpy as np

import matplotlib.pyplot as plt

x=np.arange(0.,5.,0.2)

plt.plot(x,x**4,'r',x,x*90,'bs',x,x**3,'g^')

plt.show()
```
如果要绘制多个坐标图，可以使用subpyplot命令
```python
import numpy as np

import matplotlib.pyplot as plt

x1=np.arange(0.,5.,0.2)

x2=np.arange(0.,5.,0.2)

plt.figure(1)

plt.subplot(211)

plt.plot(x1,x1**4,'r',x1,x1*90,'bs',x1,x1**3,'g^')

plt.subplot(212)

plt.plot(x2,np.cos(2*np.pi*x2),'k')

plt.show()
```
绘制直方图
```python
import numpy as np

import matplotlib.pyplot as plt

mu,sigma=100,15

x = mu+sigma*np.random.randn(1000)

n,bins,patches = plt.hist(x,10,normed=1,facecolor='g')

plt.xlabel('Frequence')

plt.ylabel('Probability')

plt.title('Example')

plt.text(40,.028,'mean=100 std.dev.=15')

plt.axis([40,160,0,0.03])

plt.grid(True)

plt.show()
```
散点图
```python
import numpy as np

import matplotlib.pyplot as plt

N=100

x = np.random.rand(N)

y = np.random.rand(N)

colors = np.random.rand(N)

area = np.pi*(10*np.random.rand(N))**2

plt.scatter(x,y,s=area,c=colors,alpha=0.5)

plt.show()
```

三维
```python
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl

from matplotlib import cm

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()

ax = fig.gca(projection='3d')

theta = np.linspace(-3*np.pi,6*np.pi,100)

z = np.linspace(-2,2,100)

r = z**2+1

x = np.sin(theta)

y = np.cos(theta)

ax.plot(x,y,z)

theta2 = np.linspace(-3*np.pi,6*np.pi,20)

z2 = np.linspace(-2,2,20)

r2 = z2**2+1

x2 = r2*np.sin(theta2)

y2 = r2*np.cos(theta2)

ax.scatter(x2,y2,z2,c='r')

x3 = np.arange(-5,5,0.25)

y3 = np.arange(-5,5,0.25)

x3,y3 = np.meshgrid(x3,y3)

R = np.sqrt(x3**2+y3**2)

z3 =np.sin (R)

surf = 
ax.plot_surface(x3,y3,z3,rstride=1,cstride=1,cmap=cm.Greys_r,linewidth=0,antialiased=False)

ax.set_zlim(-2,2)


plt.show()
```
##### 三、Pandas
基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。pandas提供了大量快速便捷地处理数据的函数和方法。是使Python成为强大而高效的数据分析环境的重要因素之一。
##### 四、Scipy
包括统计,优化,整合,线性代数模块,傅里叶变换,信号和图像处理,常微分方程求解器
##### 五、Scikit-learn
包括最常见的机器学习的算法，例如分类、回归、聚类、降维、模型选择和预处理


