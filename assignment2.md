# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by 庄茗茵 工学院



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：PyCharm 2023.3.2 (Community Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：



##### 代码

```python
class Fraction:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def simplify(self):
        common = self.gcd(self.numerator, self.denominator)
        self.numerator //= common
        self.denominator //= common

    def __add__(self, other):
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        result = Fraction(new_numerator, new_denominator)
        result.simplify()
        return result

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

# 读取输入
input_values = input().split()
numerator1, denominator1, numerator2, denominator2 = map(int, input_values)

# 创建 Fraction 对象并进行相加操作
fraction1 = Fraction(numerator1, denominator1)
fraction2 = Fraction(numerator2, denominator2)
result = fraction1 + fraction2

# 输出结果
print(result)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-3.png)




### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：



##### 代码

```python
n, w = map(int, input().split())

candies = []
for i in range(n):
    v, weight = map(int, input().split())
    candies.append((v/weight, v, weight))

candies.sort(reverse=True)

total_value = 0.0
for i, v, weight in candies:
    if w >= weight:
        total_value += v
        w -= weight
    else:
        total_value += w * (i)
        break

print('{:.1f}'.format(total_value))

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：



##### 代码

```python
for _ in range(int(input())):
    n,m,b = map(int, input().split(' '))
    d = {}
    for i in range(n):
        t,x=map(int, input().split(' '))
        if t not in d.keys():
            d[t] = [x]
        else:
            d[t].append(x)
    for i in d.keys():
        d[i].sort(reverse=True)
        d[i] = sum(d[i][:m])
    dp = sorted(d.items())
    for i in dp:
        b -= i[1]
        if b<=0:
            print(i[0])
            break
    else:
        print('alive')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-1.png)




### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：



##### 代码

```python
a = [1]*(10**6)
a[0] = 0
for i in range(1,10**3,1):
    if a[i]==1:
        for j in range(2*i+1,10**6,i+1):
            a[j]=0

n = int(input())
l = [int(x) for x in input().split()]
for i in range(n):
    m = l[i]
    if m**0.5%1==0:
        r = int(m**0.5)
        if a[r-1]==1:
            print('YES')
        else:
            print('NO')
    else:
        print('NO')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：



##### 代码

```python
for _ in range(int(input())):
    a, b = map(int, input().split())
    s = -1
    A = list(map(lambda x: int(x) % b, input().split()))
    if sum(A) % b:
        print(a)
        continue
    for i in range(a//2+1):
        if A[i] or A[~i]:
            s = a-i-1
            break
    print(s)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-5.png)




### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：



##### 代码

```python
from math import sqrt
m,n = map(int, input().split())
primes = [1]*10001
primes[0] = 0
primes[1] = 0
primes[2] = 1
for i in range(0,10001):
    if primes[i] == 1:
        for j in range(2*i,10000,i):
            primes[j] = 0

def T_prime(a):
    gen = int(sqrt(a))
    if primes[gen] and gen**2 == a:
        return True

    return False

for _ in range(m):
    cnt = 0
    ans = 0
    fen = list(map(int,input().split()))
    for i in range(len(fen)):
        if T_prime(fen[i]):
            cnt += fen[i]
            ans += 1
        
    if ans == 0:
        print(0)
    else:
        print("%.2f" % (cnt/len(fen))) 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
经常使用的gpt之一
https://chat18.aichatos.xyz/#/chat/1710056144059




