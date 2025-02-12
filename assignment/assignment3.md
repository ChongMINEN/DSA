# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by 庄茗茵 工学院



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：PyCharm 2023.3.2 (Community Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：



##### 代码

```python
def missile_interception(nums):
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] <= nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

k = int(input())
missile_heights = list(map(int, input().split()))

result = missile_interception(missile_heights)
print(result)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：



##### 代码

```python
# 将编号为numdisk的盘子从init杆移至desti杆 
def moveOne(numDisk : int, init : str, desti : str):
    print("{}:{}->{}".format(numDisk, init, desti))

#将numDisks个盘子从init杆借助temp杆移至desti杆
def move(numDisks : int, init : str, temp : str, desti : str):
    if numDisks == 1:
        moveOne(1, init, desti)
    else: 
        # 首先将上面的（numDisk-1）个盘子从init杆借助desti杆移至temp杆
        move(numDisks-1, init, desti, temp) 
        
        # 然后将编号为numDisks的盘子从init杆移至desti杆
        moveOne(numDisks, init, desti)
        
        # 最后将上面的（numDisks-1）个盘子从temp杆借助init杆移至desti杆 
        move(numDisks-1, temp, init, desti)

n, a, b, c = input().split()
move(int(n), a, b, c)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-5.png)




**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：



##### 代码

```python
def josephus(n, p, m):
    children = list(range(1, n + 1))
    result = []
    idx = p - 1

    while len(children) > 0:
        idx = (idx + m - 1) % len(children)
        result.append(str(children.pop(idx)))

    return result


# 读取输入并解决问题
inputs = []
while True:
    n, p, m = map(int, input().split())
    if n == p == m == 0:
        break
    inputs.append((n, p, m))

for n, p, m in inputs:
    result = josephus(n, p, m)
    print(','.join(result))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-3.png)




**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：



##### 代码

```python
def average_waiting_time(n, durations):
    order = list(range(n))
    order.sort(key=lambda x: durations[x])  # 按照实验时长排序学生顺序

    total_waiting_time = 0
    cumulative_time = 0
    for i in range(n):
        student_index = order[i]
        total_waiting_time += cumulative_time
        cumulative_time += durations[student_index]

    avg_waiting_time = total_waiting_time / n
    return order, avg_waiting_time

# 读取输入
n = int(input())
durations = list(map(int, input().split()))

# 计算实验顺序和平均等待时间
order, avg_wait_time = average_waiting_time(n, durations)

# 输出结果
print(' '.join(str(x+1) for x in order))
print(f'{avg_wait_time:.2f}')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：



##### 代码

```python
n = int(input())

pairs = [i[1:-1] for i in input().split()]
distances = [ sum(map(int,i.split(','))) for i in pairs]

prices = [int(x) for x in input().split()]

# ratio = distance/price
r = []
for i in range(n):
    r.append(distances[i]/prices[i])


H = zip(r,prices)
H = sorted(H, key=lambda x: (-x[0],x[1]))

#print(H)

prices.sort()    
r.sort()

import math
if n%2 == 0:
    rank = int(n/2)
    price_sq = (prices[rank-1] + prices[rank])/2
    r_sq = (r[rank-1] + r[rank])/2
else:
    rank = math.ceil(n/2)
    price_sq = prices[rank-1]
    r_sq = r[rank-1]
    
cnt = 0
for h in H:
    if h[0]>r_sq and h[1]<price_sq:
        cnt += 1 
    
print(cnt)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-1.png)




**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：



##### 代码

```python
from collections import defaultdict

n = int(input())
d = defaultdict(list)
for _ in range(n):
    #name, para = input().strip().split('-')
    name, para = input().split('-')
    if para[-1]=='M':
        d[name].append((para, float(para[:-1])/1000) )
    else:
        d[name].append((para, float(para[:-1])))


sd = sorted(d)
#print(d)
for k in sd:
    paras = sorted(d[k],key=lambda x: x[1])
    #print(paras)
    value = ', '.join([i[0] for i in paras])
    print(f'{k}: {value}')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
尚在努力中！




