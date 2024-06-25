# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by 庄茗茵 工学院



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Window 10

Python编程环境：PyCharm 2023.3.2 (Community Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：



代码

```python
sentence = input().split()
reversed_sentence = ' '.join(sentence[::-1])
print(reversed_sentence)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：



代码

```python
M, N = map(int, input().split())
words = list(map(int, input().split()))

memory = []
count = 0

for word in words:
    if word in memory:
        continue
    elif len(memory) < M:
        memory.append(word)
    else:
        memory.pop(0)
        memory.append(word)
    count += 1

print(count)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-1.png)




### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：



代码

```python
n, k = map(int, input().split())

a = list(map(int, input().split()))
a.sort()

if k == 0:
    x = 1 if a[0] > 1 else -1
elif k == n:
    x = a[-1]
else:
    x = a[k-1] if a[k-1] < a[k] else -1

print(x)


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：



代码

```python
def construct_FBI_tree(s):
    if '0' in s and '1' in s:
        node_type = 'F'
    elif '1' in s:
        node_type = 'I'
    else:
        node_type = 'B'
    
    if len(s) > 1: 
        mid = len(s) // 2
        left_tree = construct_FBI_tree(s[:mid])
        right_tree = construct_FBI_tree(s[mid:])
        return left_tree + right_tree + node_type
    else:  
        return node_type

N = int(input())
s = input()
print(construct_FBI_tree(s))


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-3.png)




### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：



代码

```python
from collections import defaultdict, deque

# 读取小组数量
t = int(input())

# 构建小组成员字典，键为成员编号，值为所属小组编号
group_members = defaultdict(int)

# 构建小组队列，键为小组编号，值为队列
group_queues = defaultdict(deque)

# 读取每个小组的成员编号
for i in range(t):
    members = list(map(int, input().split()))
    for member in members:
        group_members[member] = i + 1

# 读取命令并执行
while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        member = int(command[1])
        group = group_members[member]
        group_queues[group].append(member)
    elif command[0] == 'DEQUEUE':
        for group_queue in group_queues.values():
            if group_queue:
                print(group_queue.popleft())
                break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：



代码

```python
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []
for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)


def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)


traversal((set(parents) - set(children)).pop())

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-5.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
会做第一题！




