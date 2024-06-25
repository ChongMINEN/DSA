# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：



代码

```python
class TreeNode:
    def __init__(self):
        self.children = []
        self.first_child = None
        self.next_sib = None


def build(seq):
    root = TreeNode()
    stack = [root]
    depth = 0
    for act in seq:
        cur_node = stack[-1]
        if act == 'd':
            new_node = TreeNode()
            if not cur_node.children:
                cur_node.first_child = new_node
            else:
                cur_node.children[-1].next_sib = new_node
            cur_node.children.append(new_node)
            stack.append(new_node)
            depth = max(depth, len(stack) - 1)
        else:
            stack.pop()
    return root, depth


def cal_h_bin(node):
    if not node:
         return -1
    return max(cal_h_bin(node.first_child), cal_h_bin(node.next_sib)) + 1


seq = input()
root, h_orig = build(seq)
h_bin = cal_h_bin(root)
print(f'{h_orig} => {h_bin}')


```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image.png)




### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：



代码

```python
class BinaryTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def build_tree(lst):
    if not lst:
        return None

    value = lst.pop()
    if value == '.':
        return None

    root = BinaryTreeNode(value)
    root.left = build_tree(lst)
    root.right = build_tree(lst)

    return root


def inorder(root):
    if not root:
        return []

    left = inorder(root.left)
    right = inorder(root.right)
    return left + [root.value] + right


def postorder(root):
    if not root:
        return []

    left = postorder(root.left)
    right = postorder(root.right)
    return left + right + [root.value]


lst = list(input())
root = build_tree(lst[::-1])
in_order_result = inorder(root)
post_order_result = postorder(root)
print(''.join(in_order_result))
print(''.join(post_order_result))

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-3.png)




### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：



代码

```python
a = []
m = []

while True:
    try:
        s = input().split()
    
        if s[0] == "pop":
            if a:
                a.pop()
                if m:
                    m.pop()
        elif s[0] == "min":
            if m:
                print(m[-1])
        else:
            h = int(s[1])
            a.append(h)
            if not m:
                m.append(h)
            else:
                k = m[-1]
                m.append(min(k, h))
    except EOFError:
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-4.png)




### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：



代码

```python
def is_valid_move(board, x, y):
    if x < 0 or y < 0 or x >= len(board) or y >= len(board[0]) or board[x][y] == 1:
        return False
    return True

def dfs(board, x, y, count):
    if x < 0 or y < 0 or x >= len(board) or y >= len(board[0]) or board[x][y] == 1:
        return 0
    if count == len(board) * len(board[0]):
        return 1
    
    board[x][y] = 1
    total_paths = 0
    moves = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
    for move in moves:
        new_x, new_y = x + move[0], y + move[1]
        if is_valid_move(board, new_x, new_y):
            total_paths += dfs(board, new_x, new_y, count + 1)
    board[x][y] = 0
    return total_paths

def count_paths(n, m, x, y):
    board = [[0 for _ in range(m)] for _ in range(n)]
    return dfs(board, x, y, 1)

# 读取输入
T = int(input())
for _ in range(T):
    n, m, x, y = map(int, input().split())
    print(count_paths(n, m, x, y))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-1.png)




### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：



代码

```python
from collections import deque

def construct_graph(words):
    graph = {}
    for word in words:
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern not in graph:
                graph[pattern] = []
            graph[pattern].append(word)
    return graph

def bfs(start, end, graph):
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        word, path = queue.popleft()
        if word == end:
            return path
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern in graph:
                neighbors = graph[pattern]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
    return None

def word_ladder(words, start, end):
    graph = construct_graph(words)
    return bfs(start, end, graph)

n = int(input())
words = [input().strip() for _ in range(n)]
start, end = input().strip().split()

result = word_ladder(words, start, end)

if result:
    print(' '.join(result))
else:
    print("NO")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-5.png)




### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：



代码

```python
def is_valid_move(n, visited, row, col):
    if row < 0 or col < 0 or row >= n or col >= n or visited[row][col]:
        return False
    return True

def get_valid_moves(n, visited, row, col):
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = []
    for move in moves:
        new_row, new_col = row + move[0], col + move[1]
        if is_valid_move(n, visited, new_row, new_col):
            valid_moves.append((new_row, new_col))
    return valid_moves

def knight_tour(n, sr, sc):
    visited = [[False] * n for _ in range(n)]
    visited[sr][sc] = True
    count = 1
    current_row, current_col = sr, sc
    
    while count < n * n:
        next_moves = get_valid_moves(n, visited, current_row, current_col)
        if not next_moves:
            return False
        
        next_moves.sort(key=lambda move: len(get_valid_moves(n, visited, move[0], move[1])))
        next_row, next_col = next_moves[0]
        visited[next_row][next_col] = True
        current_row, current_col = next_row, next_col
        count += 1
    
    return True

# 读取输入
n = int(input())
sr, sc = map(int, input().split())

# 检查骑士周游是否成功
if knight_tour(n, sr, sc):
    print("success")
else:
    print("fail")

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-2.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==
很有难度




