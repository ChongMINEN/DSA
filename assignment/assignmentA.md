# Assignment #A: 图论：遍历，树算及栈

Updated 2018 GMT+8 Apr 21, 2024

2024 spring, Complied by  庄茗茵 工学院



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Window 10

Python编程环境: PyCharm 2023.3.2 (Community Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：



代码

```python
def reverse_parenthesis(s):
    stack = []
    for char in s:
        if char == ')':
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
            if stack:
                stack.pop()
            stack.extend(temp)
        else:
            stack.append(char)
    return ''.join(stack)

s = input().strip()
print(reverse_parenthesis(s))

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：



代码

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder:
        return None

    root_val = preorder[0]
    root = TreeNode(root_val)

    root_index = inorder.index(root_val)
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]

    left_preorder = [val for val in preorder if val in left_inorder]
    right_preorder = [val for val in preorder if val in right_inorder]

    root.left = build_tree(left_preorder, left_inorder)
    root.right = build_tree(right_preorder, right_inorder)

    return root

def postorder_traversal(root):
    if root is None:
        return ""

    left = postorder_traversal(root.left)
    right = postorder_traversal(root.right)

    return left + right + root.val

# 读取输入
import sys
for line in sys.stdin:
    preorder, inorder = line.strip().split()

    # 构建二叉树
    root = build_tree(preorder, inorder)

    # 后序遍历
    result = postorder_traversal(root)

    # 输出结果
    print(result)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-1.png)




### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：



代码

```python
from collections import deque

def find_multiple(n):
    if n == 1:
        return 1
    
    visited = set()
    queue = deque([(1, '1')])

    while queue:
        num, binary = queue.popleft()
        
        if num % n == 0:
            return int(binary)
        
        next_num_0 = (num * 10) % n
        next_num_1 = (num * 10 + 1) % n
        
        if next_num_0 not in visited:
            visited.add(next_num_0)
            queue.append((next_num_0, binary + '0'))
        
        if next_num_1 not in visited:
            visited.add(next_num_1)
            queue.append((next_num_1, binary + '1'))

while True:
    n = int(input())
    if n == 0:
        break
    print(find_multiple(n))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



思路：



代码

```python
from collections import deque

def min_time_to_reach_target(grid, start_row, start_col, chakra):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([(start_row, start_col, chakra, 0)])
    visited = {(start_row, start_col, chakra)}

    while queue:
        row, col, chakra, time = queue.popleft()

        if grid[row][col] == '+':
            return time

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                if (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    if grid[new_row][new_col] == '#':
                        if chakra > 0:
                            queue.append((new_row, new_col, chakra - 1, time + 1))
                    elif grid[new_row][new_col] == '*':
                        queue.append((new_row, new_col, chakra, time + 1))

    return -1

# 读取输入
M, N, T = map(int, input().split())
grid = [list(input()) for _ in range(M)]

# 找到鸣人和佐助的位置
for i in range(M):
    for j in range(N):
        if grid[i][j] == '@':
            start_row, start_col = i, j
        elif grid[i][j] == '+':
            target_row, target_col = i, j

# 计算最少时间
result = min_time_to_reach_target(grid, start_row, start_col, T)
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-5.png)




### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/



思路：



代码

```python
import heapq
m,n,p=map(int,input().split())
martix=[list(input().split())for i in range(m)]
dir=[(-1,0),(1,0),(0,1),(0,-1)]
for _ in range(p):
    sx,sy,ex,ey=map(int,input().split())
    if martix[sx][sy]=="#" or martix[ex][ey]=="#":
        print("NO")
        continue
    vis,heap,ans=set(),[],[]
    heapq.heappush(heap,(0,sx,sy))
    vis.add((sx,sy,-1))
    while heap:
        tire,x,y=heapq.heappop(heap)
        if x==ex and y==ey:
            ans.append(tire)
        for i in range(4):
            dx,dy=dir[i]
            x1,y1=dx+x,dy+y
            if 0<=x1<m and 0<=y1<n and martix[x1][y1]!="#" and (x1,y1,i) not in vis:
                t1=tire+abs(int(martix[x][y])-int(martix[x1][y1]))
                heapq.heappush(heap,(t1,x1,y1))
                vis.add((x1,y1,i))
    print(min(ans) if ans else "NO")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-3.png)




### 05442: 兔子与星空

Prim, http://cs101.openjudge.cn/practice/05442/



思路：



代码

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
还在理解代码中




