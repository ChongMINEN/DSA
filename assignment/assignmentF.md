# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

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

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：



代码

```python
from collections import deque

def right_view(n, tree):
    queue = deque([(1, tree[1])])  # start with root node
    right_view = []

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node, children = queue.popleft()
            if children[0] != -1:
                queue.append((children[0], tree[children[0]]))
            if children[1] != -1:
                queue.append((children[1], tree[children[1]]))
        right_view.append(node)

    return right_view

n = int(input())
tree = {1: [-1, -1] for _ in range(n+1)}  # initialize tree with -1s
for i in range(1, n+1):
    left, right = map(int, input().split())
    tree[i] = [left, right]

result = right_view(n, tree)
print(' '.join(map(str, result)))

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：



代码

```python
n = int(input())
a = list(map(int, input().split()))
stack = []

#f = [0]*n
for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        #f[stack.pop()] = i + 1
        a[stack.pop()] = i + 1


    stack.append(i)

while stack:
    a[stack[-1]] = 0
    stack.pop()

print(*a)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-1.png)




### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：



代码

```python
from collections import defaultdict

def dfs(node, color):
    color[node] = 1
    for neighbour in graph[node]:
        if color[neighbour] == 1:
            return True
        if color[neighbour] == 0 and dfs(neighbour, color):
            return True
    color[node] = 2
    return False

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        x, y = map(int, input().split())
        graph[x].append(y)
    color = [0] * (N + 1)
    is_cyclic = False
    for node in range(1, N + 1):
        if color[node] == 0:
            if dfs(node, color):
                is_cyclic = True
                break
    print("Yes" if is_cyclic else "No")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-6.png)




### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：



代码

```python
def canAllocateBudget(arr, budget, m):
    total = 0
    count = 0
    for x in arr:
        if total + x <= budget:
            total += x
        else:
            count += 1
            total = x
    count += 1
    return count <= m

def findMinBudget(arr, m):
    left = max(arr)
    right = sum(arr)
    result = 0
    while left <= right:
        mid = (left + right) // 2
        if canAllocateBudget(arr, mid, m):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

# 读取输入
n, m = map(int, input().split())
arr = []
for _ in range(n):
    arr.append(int(input()))

# 获取结果
result = findMinBudget(arr, m)

# 输出结果
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：



代码

```python
import heapq

def dijkstra(g):
    while pq:
        dist,node,fee = heapq.heappop(pq)
        if node == n-1 :
            return dist
        for nei,w,f in g[node]:
            n_dist = dist + w
            n_fee = fee + f
            if n_fee <= k:
                dists[nei] = n_dist
                heapq.heappush(pq,(n_dist,nei,n_fee))
    return -1

k,n,r = int(input()),int(input()),int(input())
g = [[] for _ in range(n)]
for i in range(r):
    s,d,l,t = map(int,input().split())
    g[s-1].append((d-1,l,t)) #node,dist,fee

pq = [(0,0,0)] #dist,node,fee
dists = [float('inf')] * n
dists[0] = 0
spend = 0

result = dijkstra(g)
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：



代码

```python
class Animal:
    def __init__(self, n):
        self.parent = list(range(n * 3 + 1))
    
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        else:
            self.parent[root_x] = root_y
            return True

def count_false_statements(N, K, statements):
    animal = Animal(N)
    false_count = 0
    
    for d, x, y in statements:
        if x > N or y > N or (d == 2 and x == y):
            false_count += 1
        else:
            if d == 1:
                if animal.find(x) == animal.find(y + N) or animal.find(x) == animal.find(y + 2 * N):
                    false_count += 1
                else:
                    animal.union(x, y)
                    animal.union(x + N, y + N)
                    animal.union(x + 2 * N, y + 2 * N)
            else:
                if animal.find(x) == animal.find(y) or animal.find(x) == animal.find(y + 2 * N):
                    false_count += 1
                else:
                    animal.union(x, y + N)
                    animal.union(x + N, y + 2 * N)
                    animal.union(x + 2 * N, y)
    
    return false_count

# 读取输入
N, K = map(int, input().split())
statements = [list(map(int, input().split())) for _ in range(K)]

# 输出结果
print(count_false_statements(N, K, statements))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-5.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
非常有难度，害怕即将到来的考试，努力复习中




