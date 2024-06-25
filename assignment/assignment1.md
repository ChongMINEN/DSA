# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by 庄茗茵 工学院



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

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

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：



##### 代码

```python
def tribonacci(n):
    if n == 0:
        return 0
    elif n <= 2:
        return 1
    trib = [0, 1, 1] + [0] * (n - 2)
    for i in range(3, n + 1):
        trib[i] = trib[i - 1] + trib[i - 2] + trib[i - 3]
    return trib[n]

n = int(input())
print(tribonacci(n))

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-5.png)




### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：



##### 代码

```python
def check_hello(word):
    hello = "hello"
    i = 0
    j = 0
    for i in range(len(word)):
        if j < len(hello) and word[i] == hello[j]:
            j += 1
    return j == len(hello)

word = input()

if check_hello(word):
    print("YES")
else:
    print("NO") 

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：



##### 代码

```python
def process_string(input_string):
    vowels = ["a", "o", "y", "e", "u", "i"]
    result = ""

    for char in input_string:
        char_lower = char.lower()
        if char_lower in vowels:
            continue
        else:
            result += "." + char_lower
    return result


input_string = input()
output_string = process_string(input_string)

if process_string(input_string):
    print(output_string)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-3.png)




### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：



##### 代码

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_primes_with_sum(target_sum):
    for a in range(2, target_sum):
        b = target_sum - a
        if is_prime(a) and is_prime(b):
            return a, b

# 读取输入
target_sum = int(input())

# 调用函数并输出结果
result_a, result_b = find_primes_with_sum(target_sum)
print(result_a, result_b)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：



##### 代码

```python
s = input().split('+')
res = 0
for i in s:
        pos = i.find('^')
        if pos == -1:
                continue

        if pos!=1 and int(i[:pos-1]) == 0:
                continue

        res = max(res, int(i[pos+1:]))

print(f"n^{res}")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-1.png)




### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：



##### 代码

```python
from collections import defaultdict

# 读取输入并转换成整数列表
votes = list(map(int, input().split()))

# 使用字典统计每个选项的票数
vote_counts = defaultdict(int)
for vote in votes:
    vote_counts[vote] += 1

# 找出得票最多的票数
max_votes = max(vote_counts.values())

# 按编号顺序收集得票最多的选项
winners = sorted([item for item in vote_counts.items() if item[1] == max_votes])

# 输出得票最多的选项，如果有多个则并列输出
print(' '.join(str(winner[0]) for winner in winners))


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==
基础很差，依旧让gpt解释每行代码，这样比较能理解。




