import copy
import collections

nums = [3,5,1,9,2,4,6,8]

## 快速排序
def partition(nums, p, q):
    x = nums[p]
    i = p
    for j in range(p+1, q+1):
        if nums[j] <= x:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[p], nums[i] = nums[i], nums[p]
    return i

def quicksort(nums, left, right):
    if left < right:
        mid = partition(nums, left, right)
        quicksort(nums, left, mid-1)
        quicksort(nums, mid+1, right)
    return nums

# print(quicksort(nums, 0, len(nums)-1))

## 归并排序
def merge(l, r):
    ans = []
    i, j = 0, 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]:
            ans.append(l[i])
            i += 1
        else:
            ans.append(r[j])
            j += 1
    if i == len(l):
        ans += r[j:]
    else:
        ans += l[i:]
    return ans

def mergesort(nums):
    if len(nums) <= 1:
        return nums
    return merge(mergesort(nums[:len(nums)//2]), mergesort(nums[len(nums)//2:]))

# print(mergesort(nums))

## 堆排序
def heapajust(R, top, bottom):
    rc = R[top]
    while top*2+1 <= bottom:
        j = top*2+1
        if j < bottom and R[j] < R[j+1]:
            j += 1
        if rc >= R[j]:
            break
        R[top] = R[j]
        top = j
    R[top] = rc

def heapsort(nums):
    n = len(nums)
    for i in range(n//2)[::-1]:
        heapajust(nums, i, n-1)
    for i in range(1, n)[::-1]:
        nums[0], nums[i] = nums[i], nums[0]
        heapajust(nums, 0, i-1)
    return nums

# print(heapsort(nums))

## 计数排序
def countingsort(nums, k):
    c = [0]*(k+1)
    b = [0]*len(nums)
    for i in range(len(nums)):
        c[nums[i]] += 1
    for i in range(1, k+1):
        c[i] += c[i-1]
    for i in range(len(nums))[::-1]:
        b[c[nums[i]]-1] = nums[i]
        c[nums[i]] -= 1
    return b

# print(countingsort([2,4,2,5,3,1,6], 6))

## 随机选择，找出数组中第k小的数
def random_select(nums, p, q, k):
    if p == q:
        return nums[p]
    r = partition(nums, p, q)
    i = r-p+1   # 这里算的是r在nums[p:q+1]序列中排第几
    if i == k:
        return nums[r]
    if k < i:
        random_select(nums, p, r-1, k)
    else:
        random_select(nums, r+1, q, k-i)
# print(random_select(nums, 0, len(nums)-1, 3))

# 二叉树结点
class ListNode:
    def __init__(self, value = 0):
        self.value = value
        self.leftnode = None
        self.rightnode = None

# 对二叉树进行遍历、复制、计算深度、计算节点数、计算叶子节点数
class BinaryTree:
    # 建立二叉树
    def buildBiTree(self, T):
        p = input('Please input something')
        if p == '#':
            T = None
        else:
            T = ListNode(p)
            T.leftnode = self.buildBiTree(T.leftnode)
            T.rightnode = self.buildBiTree(T.rightnode)
        return T

    # 中序遍历非递归算法
    def inorderTraverse(self, T):
        p = T
        stack = []
        while p is not None or len(stack) != 0:
            if p is not None:
                stack.append(p)
                p = p.leftnode
            else:
                p = stack.pop()
                print(p.value)
                p = p.rightnode

    # 层次遍历
    def levelorder(self, T):
        queue = collections.deque()
        queue.append(T)
        while len(queue) > 0:
            p = queue.popleft()
            print(p.value)
            if p.leftnode is not None:
                queue.append(p.leftnode)
            if p.rightnode is not None:
                queue.append(p.rightnode)
    # 复制二叉树
    def Copy(self, T):
        if T is None:
            new_T = None
        else:
            new_T = ListNode(T.value)
            new_T.leftnode = self.Copy(T.leftnode)
            new_T.rightnode = self.Copy(T.rightnode)
        return new_T

    # 计算二叉树的深度
    def Depth(self, T):
        if T is None:
            return 0
        return max(self.Depth(T.leftnode), self.Depth(T.rightnode))+1

    # 计算二叉树结点数
    def NodeCount(self, T):
        if T is None:
            return 0
        return self.NodeCount(T.leftnode) + self.NodeCount(T.rightnode) + 1

    # 计算二叉树叶子结点数
    def LeafNode(self, T):
        if T is None:
            return 0
        if T.leftnode is None and T.rightnode is None:
            return 1
        return self.LeafNode(T.leftnode) + self.LeafNode(T.rightnode)

# if __name__ == '__main__':
#     BiTree = BinaryTree()
#     tree = BiTree.buildBiTree(None)
#     BiTree.levelorder(tree)
#     BiTree.inorderTraverse(tree)
#     print(BiTree.Copy(tree))
#     print(BiTree.Depth(tree))
#     print(BiTree.NodeCount(tree))
#     print(BiTree.LeafNode(tree))


# 二叉排序（查找）树
class OperationTree:
    ## 插入数据，生成二叉树
    def insert(self, root, value):
        if root == None:
            root = ListNode(value)
        elif root.value > value:
            root.leftnode = self.insert(root.leftnode, value)
        else:
            root.rightnode = self.insert(root.rightnode, value)
        return root

    ## 非遍历方式查找
    def quary_non(self, root, value):
        while root:
            if root.value == value:
                return True
            elif root.value < value:
                root = root.rightnode
            else:
                root = root.leftnode
        return False

    ## 遍历方式查找
    def quary_con(self, root, value):
        if root == None:
            return False
        if root.value == value:
            return True
        elif root.value < value:
            return self.quary_con(root.rightnode, value)
        else:
            return self.quary_con(root.leftnode, value)

    ## 寻找最小值
    def findmin(self, root):
        if root.leftnode != None:
            return self.findmin(root.leftnode)
        else:
            return root

    ## 寻找最大值
    def findmax(self, root):
        if root.rightnode != None:
            return self.findmax(root.rightnode)
        else:
            return root

    ## 中序遍历，也就是BST排序
    def mid_traverse(self, root):
        if root != None:
            self.mid_traverse(root.leftnode)
            print(root.value, end = ' ')
            self.mid_traverse(root.rightnode)

    ## 删除节点
    def delNode(self, root, value):
        if root == None:
            return
        if root.value < value:
            root.rightnode = self.delNode(root.rightnode, value)
        elif root.value > value:
            root.leftnode = self.delNode(root.leftnode, value)
        else:
            if root.leftnode and root.rightnode:
                temp = self.findmax(root.leftnode)
                root.value = temp.value
                root.leftnode = self.delNode(root.leftnode, temp.value)
            elif root.leftnode == None and root.rightnode == None:
                root = None
            elif root.leftnode == None:
                root = root.rightnode
            elif root.rightnode == None:
                root = root.leftnode
        return root
# if __name__ == '__main__':
#     ## 实例化
#     ot = OperationTree()
#     tree = None
#     ## 插入数据
#     for i in nums:
#         tree = ot.insert(tree, i)
#     print('中序遍历并打印tree')
#     ot.mid_traverse(tree)
#     print('\n')
#     print('非遍历方式查找9', ot.quary_non(tree, 9))
#     print('遍历方式查找5', ot.quary_con(tree, 5))
#     print('最小值为', ot.findmin(tree).value)
#     print('最大值为', ot.findmax(tree).value)
#     print('删除5', ot.delNode(tree, 5))
#     print('中序遍历并打印tree')
#     ot.mid_traverse(tree)


# 二叉平衡树
class AVLTree:
    # 返回左子树的高度
    def left_height(self, node):  # 开始传入根结点，后面传入每颗子树的根结点
        if node is None:
            return 0
        return self.tree_height(node.leftnode)

    # 返回右子树的高度
    def right_height(self, node):  # 开始传入根结点，后面传入每颗子树的根结点
        if node is None:
            return 0
        return self.tree_height(node.rightnode)

    # 返回以该结点为根结点的树的高度
    def tree_height(self, node):
        if node is None:
            return 0
        return max(self.tree_height(node.leftnode), self.tree_height(node.rightnode)) + 1

    #
    def left_rotation(self, node):
        # 创建新的节点用来保存左旋后的节点
        new_node = copy.deepcopy(node)
        new_node.rightnode = node.rightnode
        new_node.leftnode = node.leftnode.rightnode
        # 连接新的左右节点
        node.value = node.leftnode.value
        node.leftnode = node.leftnode.leftnode
        node.rightnode = new_node

    #
    def right_rotation(self, node):
        # 创建新的节点用来保存左旋后的节点
        new_node = copy.deepcopy(node)
        new_node.leftnode = node.leftnode
        new_node.rightnode = node.rightnode.leftnode
        # 连接新的左右节点
        node.value = node.rightnode.value
        node.rightnode = node.rightnode.rightnode
        node.leftnode = new_node

    # 添加节点
    def insert(self, root, value):
        if root == None:
            root = ListNode(value)
        elif root.value > value:
            root.leftnode = self.insert(root.leftnode, value)
        else:
            root.rightnode = self.insert(root.rightnode, value)
        return root

    # 判断二叉排序树是否需要调整（是否达到平衡）
    def judge_node(self, node):
        # 如果右子树高于左子树
        if self.right_height(node) - self.left_height(node) > 1:
            # 如果右子树的左子树高度大于右子树的右子树高度，则为RL型
            if node.rightnode and self.left_height(node.rightnode) > self.right_height(node.rightnode):
                # 先对当前结点的右子节点进行右旋转
                self.right_rotation(node.rightnode)
                # 再对当前节点进行左旋
                self.left_height(node)
            # 如果右子树的右子树高度大于右子树的左子树高度，则为RR型
            else:
                # 直接进行左旋转
                self.left_rotation(node)
            return
        if self.left_height(node) - self.right_height(node) > 1:
            if node.left and self.right_height(node.leftnode) > self.left_height(node.leftnode):
                self.left_rotation(node.left)
                self.right_rotation(node)
            else:
                self.right_rotation(node)
            return

    ## 利用后序遍历完成树的平衡
    def backward_traverse(self, root):
        if root != None:
            self.backward_traverse(root.leftnode)
            self.backward_traverse(root.rightnode)
            self.judge_node(root)

    ## 中序遍历，也就是BST排序
    def mid_traverse(self, root):
        if root != None:
            self.mid_traverse(root.leftnode)
            print(root.value, end=' ')
            self.mid_traverse(root.rightnode)

    ## 前序遍历，也就是BST排序
    def forward_traverse(self, root):
        if root != None:
            print(root.value, end=' ')
            self.forward_traverse(root.leftnode)
            self.forward_traverse(root.rightnode)

# if __name__ == '__main__':
#     avltree = AVLTree()
#     tree = None
#     for i in nums:
#         tree = avltree.insert(tree, i)
#         avltree.backward_traverse(tree)
#     print(avltree.mid_traverse(tree))
#     print(avltree.forward_traverse(tree))


# 红黑树结点
class RBTree_Node:
    def __init__(self, key, right, left, p, color):
        self.key = key
        self.right = right
        self.left = left
        self.p = p
        self.color = color

# 红黑树
class RBTree:
    def __init__(self, root, nil):
        self.root = root
        self.nil = nil

    # 简单的插入操作，类似二叉查找树，这里的z已经是一个结点，而不只是一个数值
    def tree_insert(self, z):
        y = self.nil
        x = self.root
        while x!=self.nil:
            # 保留x的上一个结点
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        # 如果根本没进入循环，这时y还是self.nil
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = 'red'
        self.rb_insert_fixup(z)

    # 左旋
    def left_rotation(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y

    # 右旋
    def right_rotation(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.nil:
            x.right.p = y
        x.p = y.p
        if y.p == self.nil:
            self.root = x
        elif y.p.left == y:
            y.p.left = x
        else:
            y.p.right = x
        x.right = y
        y.p = x

    # 红黑树的插入
    def rb_insert_fixup(self, z):
        # 当前结点的父亲结点也是红色时就违反了规则
        while z.p.color == 'red':
            # 当前结点的父节点在其祖父结点的左边时
            if z.p == z.p.p.left:
                # 将y记为当前结点的叔叔结点
                y = z.p.p.right
                # 如果叔叔节点的颜色是红色，就将父节点和叔叔结点都变成黑色，将祖父结点变成红色，再将祖父节点设成当前结点进行迭代
                if y.color == 'red':
                    z.p.color = 'black'
                    z.p.p.color = 'red'
                    y.color = 'black'
                    z = z.p.p
                # 如果叔叔结点的颜色是黑色，就判断当前结点在其父节点的左边还是右边
                else:
                    # 如果当前结点在其父节点的右边，就对其父节点右旋，从而归纳到在其父节点左边这种情况
                    if z == z.p.right:
                        z = z.p
                        self.left_rotation(z)
                    # 如果当前结点在其父节点的左边， 就对其祖父节点右旋， 再交换其父节点和祖父节点的颜色
                    z.p.color = 'black'
                    z.p.p.color = 'red'
                    self.right_rotation(z.p.p)
            # 当前结点的父节点在其祖父结点的右边时
            else:
                y = z.p.p.left
                if y.color == 'red':
                    z.p.color = 'black'
                    y.color = 'black'
                    z.p.p.color = 'red'
                    z = z.p.p
                else:
                    if z == z.left:
                        z = z.p
                        self.right_rotation(z)
                    z.p.color = 'black'
                    z.p.p.color = 'red'
                    self.left_rotation(z.p.p)
        self.root.color = 'black'

    # 中序遍历
    def mid_traverse(self, x):
        if x != self.nil:
            self.mid_traverse(x.left)
            print(x.key)
            self.mid_traverse(x.right)

    # 查找
    def search(self, x, k):
        if x == self.nil:
            return False
        if k == x.key:
            return True
        if k < x.key:
            return self.search(x.left, k)
        else:
            return self.search(x.right, k)

# if __name__ == '__main__':
#
#     nil = RBTree_Node(0, None, None, None, 'black')
#     root = RBTree_Node(7, nil, nil, nil, 'black')
#     t = RBTree(root, nil)
#     for i in nums:
#         z = RBTree_Node(i, None, None, None, 'red')
#         t.tree_insert(z)
#     t.mid_traverse(t.root)


# 字符串匹配BF算法
class BF:
    # 返回p在str中匹配的第一个字符的下标
    def string_match(self, strs, p):
        n, m = len(strs), len(p)
        i, j = 0, 0
        while i < n and j < m:
            if strs[i] != p[j]:
                i = i-j+1
                j = 0
            else:
                i += 1
                j += 1
        if j >= m:
            return i-m
        else:
            return -1

#字符串匹配KMP算法
class KMP:
    def get_nextval(self, needle):
        i, j = 0, -1
        nextval=[-1]*len(needle)
        while i < len(needle)-1:
            if j == -1 or needle[i] == needle[j]:
                i += 1
                j += 1
                if needle[i] != needle[j]:
                    nextval[i] = j
                else:
                    nextval[i] = nextval[j]
            else:
                j = nextval[j]
        return nextval

    def strStr(self, haystack: str, needle: str) -> int:
        nextval = self.get_nextval(needle)
        i, j = 0, 0
        while i <= len(haystack)-1 and j <= len(needle)-1:
            if j == -1 or haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                j = nextval[j]
        if j >= len(needle):
            return i-j
        else:
            return -1

# if __name__ == '__main__':
#     sm = KMP()
#     print(sm.strStr('dndvnsfivnsidpf', 'vns'))


# Huffman Tree
class HTnode:
    def __init__(self, weight = 0):
        self.weight = weight
        self.parent = None
        self.leftnode = None
        self.rightnode = None

    def is_left(self):
        return self.parent.leftnode == self

class HaffmanTree:
    # 统计字符出现频率，生成映射表char_frequency
    def count_frequency(self, text):
        chars = []
        ret = []
        for char in text:
            if char in chars:
                continue
            else:
                chars.append(char)
                ret.append((char, text.count(char)))
        return ret

    # 生成和char_ferquency顺序一致的结点的列表nodes
    def create_nodes(self, frequency_list):
        # 传入[item[1] for item in char_frequency]
        return [HTnode(frequency) for frequency in frequency_list]

    # 创建哈夫曼树
    def createHaffmanTree(self, nodes):
        queue = nodes[:]
        # 直到最后只剩一个根节点
        while len(queue) > 1:
            queue.sort(key=lambda item:item.weight)
            node_left = queue.pop(0)
            node_right = queue.pop(0)
            node_parent = HTnode(node_left.weight + node_right.weight)
            node_parent.leftnode = node_left
            node_parent.rightnode = node_right
            node_left.parent = node_parent
            node_right.parent = node_parent
            queue.append(node_parent)

        return queue[0]

    # Huffman编码
    # 生成和nodes顺序一致的密码表
    def huffman_encoding(self, nodes, root):
        #传入create_nodes创造的结点列表和哈夫曼树的根节点
        huffman_code = ['']*len(nodes)
        for i in range(len(nodes)):
            node = nodes[i]
            while node != root:
                if node.is_left():
                    huffman_code[i] = '0'+huffman_code[i]
                else:
                    huffman_code[i] = '1'+huffman_code[i]
                node = node.parent
        return huffman_code

    # 编码整个字符串
    # 哈夫曼树中只有权重，没有对应字符，同样nodes表和codes表中都没有字符
    # 只能先在字符频数映射表char_frequency中找到对应字符的坐标，再对应到codes的坐标找到对应密码
    def encode_str(self, text, char_frequency, codes):
        # 传入待编码的字符串，count_frequency创造的字符和频率的列表，和huffman_encoding创造的每个字符对应的编码
        ret = ''
        for char in text:
            i = 0
            for item in char_frequency:
                if char == item[0]:
                    ret += codes[i]
                # 如果char和item[0]不相等就让i加1， 直到相等的时候i就是char对应的codes的下标
                i += 1
        return ret

    # 解码整个字符串
    # 因为没有一个叶子结点是另一个叶子结点的祖先，所以每个叶结点的编码就不可能是其它叶结点编码的前缀
    # 这也是可以使用长度不等的二进制数进行编码而不用担心解码混乱的原因
    # 先在codes表中找到对应密码的坐标，再按照坐标在char_frequency表中找到对应字符
    def decode_str(self, huffman_str, char_frequency, codes):
        ret = ''
        while huffman_str != '':
            i = 0
            for item in codes:
                if item in huffman_str and huffman_str.index(item) == 0:
                    ret += char_frequency[i][0]
                    huffman_str = huffman_str[len(item):]
                i += 1
        return ret

# if __name__ == '__main__':
#     text = input('The text to encode:')# 该函数将所有输入当作字符串看待
#     ht = HaffmanTree()
#     char_frequency = ht.count_frequency(text)
#     nodes = ht.create_nodes([item[1] for item in char_frequency])
#     root = ht.createHaffmanTree(nodes)
#     codes = ht.huffman_encoding(nodes, root)
#
#     huffman_str = ht.encode_str(text, char_frequency, codes)
#     origin_str = ht.decode_str(huffman_str, char_frequency, codes)
#
#     print(codes)
#     print('Encoder result:' + huffman_str)
#     print('Decode result:' + origin_str)


# 定义一个图类型，包含一个顶点表和一个邻接矩阵
class ALGraph_matrix:
    def __init__(self, vexs, arcs):
        # 传入总顶点数vexnum，总边数arcnum
        self.vexnum = len(vexs)
        self.arcnum = len(arcs)
        self.vexs = vexs
        self.arcs = [[0] * self.vexnum for _ in range(self.vexnum)]

# 用邻接矩阵表示法创建无向网
class UDC_matrix:
    # 在顶点表中找顶点的下标
    def LocateVex(self, G, u):
        for i in range(G.vexnum):
            if u == G.vexs[i]:
                return i
    # 建立无向网的邻接矩阵
    def createUDN(self, G, arcs):
        maxint = 2**31-1
        # 初始化邻接矩阵
        for i in range(G.vexnum):
            for j in range(G.vexnum):
                G.arcs[i][j] = maxint
        # 构建邻接矩阵

        for k in range(G.arcnum):
            v1,v2,w = arcs[k]
            i = self.LocateVex(G, v1)
            j = self.LocateVex(G, v2)
            G.arcs[i][j] = int(w)
            # G.arcs[j][i] = G.arcs[i][j]

# if __name__ == '__main__':
#     vexs = ['v1','v2','v3','v4','v5','v6','v7','v8','v9']
#     arcs = [('v1','v2',6),('v1','v3',4),('v1','v4',5),('v2','v5',1),('v3','v5',1),('v4','v6',2),
#             ('v5','v7',9),('v5','v8',7),('v6','v8',4),('v7','v9',2),('v8','v9',4)]
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)


# 定义弧（边）的存储结构
class arcNode:
    def __init__(self, adjvex, weight):
        self.adjvex = adjvex
        self.nextarc = None
        self.weight = weight

# 定义一个顶点的存储结构
class vexNode:
    def __init__(self, vex):
        self.vex = vex
        self.firstarc = None
# 定义一个图的类型，包含一个特殊的列表

class ALGraph_table:
    def __init__(self, vexs, arcs):
        self.vexnum = len(vexs)
        self.arcnum = len(arcs)
        self.vertices = []

# 用邻接表表示法创建无向网
class UDG_table:
    def LocateVex(self, G, u):
        for i in range(G.vexnum):
            if u == G.vertices[i].vex:
                return i
    def createUDG(self, G, vexs, arcs):
        for i in range(G.vexnum):
            vexnode = vexNode(vexs[i])
            G.vertices.append(vexnode)
        for j in range(G.arcnum):
            v1, v2, w = arcs[j]
            i = self.LocateVex(G, v1)
            j = self.LocateVex(G, v2)
            # 使用头插法将新结点p1插入到顶点v_i的边表头部
            p1 = arcNode(j, int(w))
            p1.nextarc = G.vertices[i].firstarc
            G.vertices[i].firstarc = p1
            # 因为是无向网，所以要生成另一个对称的新的边界点p2
            p2 = arcNode(i, int(w))
            p2.nextarc = G.vertices[j].firstarc
            G.vertices[j].firstarc = p2

# if __name__=='__main__':
#     vexs = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
#     arcs = [('v1', 'v2', 6), ('v1', 'v3', 4), ('v1', 'v4', 5), ('v2', 'v5', 1), ('v3', 'v5', 1), ('v4', 'v6', 2),
#             ('v5', 'v7', 9), ('v5', 'v8', 7), ('v6', 'v8', 4), ('v7', 'v9', 2), ('v8', 'v9', 4)]
#     G = ALGraph_table(vexs, arcs)
#     udg = UDG_table()
#     udg.createUDG(G, vexs, arcs)

# 图在邻接矩阵上的的深度优先遍历（dfs）
class dfs_matrix:
    # 创建辅助数组
    def __init__(self, G):
        self.visited = [False]*G.vexnum
        self.maxint = 2**31-1
    def dfs(self, G, v):
        # 传入图结构和首先遍历的结点
        print(v)
        for i in range(G.vexnum):
            if v == G.vexs[i]:
                break
        self.visited[i] = True
        for k in range(G.vexnum):
            if G.arcs[i][k] != self.maxint and not self.visited[k]:
                self.dfs(G, G.vexs[k])

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     dfs_m = dfs_matrix(G)
#     # 第一个位置输入一个图结构，第二个位置输入你想开始的顶点
#     dfs_m.dfs(G, 'a')

# 图在邻接表上的的深度优先遍历（dfs）
class dfs_table:
    # 创建辅助数组
    def __init__(self, G):
        self.visited = [False]*G.vexnum
    def dfs(self, G, v):
        # 传入图结构和首先要遍历的结点
        print(v)
        # 在图的结点表G.vexnum中找到v所对应的下标，需要保证输入的v一定是G的一个顶点
        for i in range(G.vexnum):
            if v == G.vertices[i].vex:
                break
        self.visited[i] = True
        # 指针p指到顶点的下一个结点
        p = G.vertices[i].firstarc
        while p != None:
            if not self.visited[p.adjvex]:
                # p.adjvex代表的是邻结点的下标，需要去G.vertices表中找到该结点所表示的字符
                self.dfs(G, G.vertices[p.adjvex].vex)
            p = p.nextarc

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_table(vexs, arcs)
#     udg = UDG_table()
#     udg.createUDG(G, vexs, arcs)
#     dfs_t = dfs_table(G)
#     dfs_t.dfs(G, 'a')

# 图在邻接表上的广度优先非递归遍历
class bfs_table:
    def __init__(self, G):
        self.visited = [False]*G.vexnum
    def bfs(self, G, v):
        print(v)
        # 在图的顶点表中找到v的下标
        for i in range(G.vexnum):
            if v == G.vertices[i].vex:
                break
        self.visited[i] = True
        # 建立一个队列
        queue = []
        # 在队列中加入顶点在顶点表中的下标
        queue.append(i)
        while len(queue) != 0:
            # 让p指向出队的顶点的下一个结点
            p = G.vertices[queue.pop(0)].firstarc
            while p != None:
                # 如果这个结点没有被访问过
                if not self.visited[p.adjvex]:
                    print(G.vertices[p.adjvex].vex)
                    self.visited[p.adjvex] = True
                    # 将这个结点的下标存入队列
                    queue.append(p.adjvex)
                p = p.nextarc

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_table(vexs, arcs)
#     udg = UDG_table()
#     udg.createUDG(G, vexs, arcs)
#     bfs_t = bfs_table(G)
#     bfs_t.bfs(G, 'a')

# 生成树：包含无向图所有顶点的极小连通子图（去掉一条边则非连通） 或 所有顶点均由边连接在一起，但不存在回路的图
# 生成树的性质：一个有n个顶点的连通图的生成树有n-1条边
#           生成树中再加一条边必然形成回路
#           生成树中任意两个顶点间的路径是唯一的
# 最小生成树：给定一个无向网络，在该网的所有生成树中，使得各边权值之和最小的那棵生成树称为该网的最小生成树
# MST(Mininum Spanning Tree)性质：设N=(V,E)是一个连通网，U是顶点集V的一个非空子集。
# 若边(u,v)是一条具有最小权值的边，其中u∈U，v∈V-U，则必存在一棵包含边(u,v)的最小生成树。

# 构造最下生成树方法一：普里姆(Prim)算法
# 使用邻接矩阵
class MiniTree_P:
    def create_mini_tree_prim(self, G, start):
        visited = [0]*G.vexnum
        for i in range(G.vexnum):
            if start == G.vexs[i]:
                break
        visited[i] = 1
        for k in range(1, G.vexnum): # 因为最小生成树一定有G.vexnum-1条边
            min_weight = float('inf') # 初始化最小权重
            for i in range(G.vexnum):
                for j in range(G.vexnum):
                    # 确保两个顶点一个来自访问过的集合，一个来自没访问过的集合，并找到最小权重
                    if visited[i] == 1 and visited[j] == 0 and G.arcs[i][j] < min_weight:
                        min_weight = G.arcs[i][j]
                        # 记录两个建立的最小生成树的顶点的下标
                        v1 = i
                        v2 = j
            print('边：%s -> %s 权值：%d '%(G.vexs[v1], G.vexs[v2], min_weight))
            visited[v2] = 1

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     MT = MiniTree_P()
#     MT.create_mini_tree_prim(G, 'a')

# 构造最小生成树方法二：克鲁斯卡尔(Kruskal)算法

class MiniTree_K:
    def __init__(self):
        self.X = dict()
        self.rank = dict() # 各点的初始等级均为0, 如果被做为连接的的末端，则增加1
    # 制作一个列表，它是由一个图的所有边组成的，第一个元素代包含两个顶点，第二个元素为该边的权值
    def edge_set(self, G):
        edges = []
        for i in range(G.vexnum):
            for j in range(i):
                edges.append((i, j, G.arcs[i][j]))
        return edges
    # 判断不构成环路的方法：将两个不连通的顶点集分别赋予不同的记号。
    # 添加新的边的时候如果两个顶点所在记号相同则会构成环，否则不会。需要在连接两顶点之后将其换为相同记号
    # 以全局变量X定义节点集合，即类似{'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
    # 如果A、B两点连通，则会更改为{'A': 'B', 'B': 'B",...},即任何两点连通之后，两点的值value将相同。
    def make_set(self,point):
        self.X[point] = point
        self.rank[point] = 0
    # 由于判断是否为环的时候需要看这两个结点的记号是否为同一个，因此需要find他们的根节点
    def find(self,point):
        if self.X[point] != point:
            self.X[point] = self.find(self.X[point])
        return self.X[point]
    # 找到当前最小权重之后需要糅合顶点所属的两部分
    def merge(self, point1, point2):
        root1 = self.find(point1)
        root2 = self.find(point2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.X[root2] = root1
            else:
                self.X[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
    def create_mini_tree_kru(self, G):
        edges = self.edge_set(G) # 得到所有有权边的信息，格式为[(i,j,weight)]
        edges.sort(key = lambda x : x[2])
        for vex in G.vexs:
            self.make_set(vex)
        mini_tree = []
        for edge in edges:
            vex1, vex2, weight = edge
            if self.find(G.vexs[vex1]) != self.find(G.vexs[vex2]):
                self.merge(G.vexs[vex1], G.vexs[vex2])
                mini_tree.append(edge)
        print(mini_tree)
        return mini_tree

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     MT = MiniTree_K()
#     MT.create_mini_tree_kru(G)

# 最短路径

class shortest_D:
    def LocateVex(self, G, u):
        for i in range(G.vexnum):
            if u == G.vexs[i]:
                return i

    # Dijkstra算法
    def Dijkstra(self, G, v):
        j = self.LocateVex(G, v)
        # S：已求出最短路径的顶点的集合
        S = [j]
        # T：=V-S，尚未确定最短路径的顶点集合
        T = [k for k in range(G.vexnum)]
        T.remove(j)
        # 记录每个顶点到v的当前最短距离
        D = G.arcs[j]
        while len(T) > 0:
            min_point = T[0]
            # 找到最短的距离
            for i in T:
                if D[i] < D[min_point]:
                    min_point = i
            S.append(min_point)
            # 将到v点最短距离的顶点移除
            T.remove(min_point)
            # min_point点到v点的距离已经确定
            print('v到%s的最短距离为%d' %(G.vexs[min_point], D[min_point]))
            # 计算经过最短距离点时其他顶点的距离有没有减小，如有减小则更新
            for i in T:
                if D[min_point] + G.arcs[min_point][i] < D[i]:
                    D[i] = D[min_point] + G.arcs[min_point][i]
        return D
    # Floyd算法
    # 该算法是一个解决求多源最小路径问题的算法，当然对于这个问题我们可以使用n次dijkstra算法，但是它的复杂度会是O(n^3)
    # 一开始先初始化一个矩阵，该矩阵就是将邻接矩阵的对角线都变为0
    # 逐步试着在原路径中增加中间结点，若加入后路径变短，则修改；否则，维持原值。
    # 所有顶点试探完毕，算法结束。
# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     SP = shortest_D()
#     print(SP.Dijkstra(G, 'v0'))


# AOV网：用一个有向图表示一个工程的各子工程及其相互制约的关系。
# 其中以顶点表示活动，弧表示活动之间的优先制约关系，
# 称这种有向图为顶点表示活动的网，简称AOV网(Activity ON Vertex network)
# 拓扑排序
class TS:
    def LocateVex(self, G, u):
        for i in range(G.vexnum):
            if u == G.vexs[i]:
                return i
    def TopologicalSort(self, G):
        max_int = 2**31-1
        # 创建入度字典
        in_degree = dict((u, 0) for u in G.vexs)
        # 获取每个结点的入度
        for i in range(G.vexnum):
            for j in range(G.vexnum):
                if G.arcs[i][j] < max_int:
                    in_degree[G.vexs[j]] += 1
        # 使用列表作为队列并将入度为0的添加到队列中
        Q = [u for u in G.vexs if in_degree[u] == 0]
        res = []
        # 当队列中有元素时执行
        while Q:
            # 从队列
            u = Q.pop(0)
            i = self.LocateVex(G, u)
            res.append(u)
            # 移除与取出元素相关的指向，即将所有与取出元素相关的元素的入度减少1
            for j in range(G.vexnum):
                if G.arcs[i][j] < max_int:
                    v = G.vexs[j]
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        Q.append(v)
        return res

# if __name__ == '__main__':
#     vexs =
#     arcs =
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     ts = TS()
#     print(ts.TopologicalSort(G))


# AOE网：用一个有向图表示一个工程的各子工程及其相互制约的关系。
# 以弧(ai)表示活动，以顶点(vj)表示活动的开始或结束的事件，称这种有向图为边表示活动的网，简称为AOE网(Activity On Edge)
# 问题的关键路径问题
# ve(vj)--表示事件vj的最早发生时间
# vl(vj)--表示事件vj的最迟发生时间
# e(i)--表示活动ai的最早开始时间
# l(i)--表示活动ai的最迟开始时间
# l(i)-e(i)--表示完成活动ai的时间余量
# 关键活动--关键路径上的活动，即l(i)==e(i)的活动
# 假设活动ai两端的顶点为j,k，其持续时间记为w_{j,k}
# 有：e(i)=ve(j),l(i)=vl(k)，下面需要求ve(j)和vl(j)
# ve(j)=Max_i{ve(i)+w_{i,j}}
# vl(i)=Min_j(vl(j)-w{i,j}}

# 请按拓扑排序输入顶点
def critical_path(vexs, arcs):

    '''计算各个顶点的ve(v)最早发生时间'''

    # 找出图的起点
    # start_vex= vexs[:] # 记录没有入度的点，也就是图的源点
    # for arc in arcs:
    #     if arc[1] in start_vex:
    #         start_vex.remove(arc[1])
    ve_vex_dict = {} # 用来存放每个结点的最早开始时间
    # ve_vex_dict[start_points[0]] = 0 # 随意取一个源点作为开始
    # 计算每一个顶点的最早开始时间
    # 由于vexs按拓扑排序输入，计算到某一点时它前面的点一定已经计算完了
    for vex in vexs:
        ve_time_all = []
        for arc in arcs:
            if arc[1] == vex:
                ve_time_all.append(ve_vex_dict[arc[0]] + arc[2])
        if len(ve_time_all) == 0:
            ve_vex_dict[vex] = 0
        else:
            ve_vex_dict[vex] = max(ve_time_all)
    print('ve(v)最早发生时间：\n', ve_vex_dict, '\n')

    '''计算各个顶点的vl(v)最迟发生时间'''

    # 找出图的终点
    # end_vex = vexs[:] # 记录没有出度的点也就是图的汇点
    # for arc in arcs:
    #     if arc[0] in end_vex:
    #         end_vex.remove(arc[0])
    # max_time, max_vex = 0, 0 # 记录最久的时间
    # for vex in end_vex:
    #     max_vex = vex if ve_point_dict[vex] > max_time else max_vex
    #     max_time = max(max_time, ve_vex_dict[vex])
    max_time = max(ve_vex_dict.values())
    vl_vex_dict = {} # 用来存放每个结点的最迟开始时间
    for vex in vexs[::-1]:
        vl_time_all = []
        for arc in arcs:
            if arc[0] == vex:
                vl_time_all.append(vl_vex_dict[arc[1]] - arc[2])
        if len(vl_time_all) == 0:
            vl_vex_dict[vex] = max_time
        else:
            vl_vex_dict[vex] = min(vl_time_all)
    print('vl(v)最迟发生时间：\n', vl_vex_dict, '\n')

    '''计算各个边的e(a)最早发生时间'''
    e_arc_dict = {}
    for arc in arcs:
        e_arc_dict['{}-{}'.format(arc[0], arc[1])] = ve_vex_dict[arc[0]]
    print('e(a)最早发生时间：\n',e_arc_dict,'\n')

    '''计算各个边的l(a)最迟发生时间'''
    l_arc_dict = {}
    for arc in arcs:
        l_arc_dict['{}-{}'.format(arc[0], arc[1])] = vl_vex_dict[arc[1]] - arc[2]
    print('l(a)最迟发生时间：\n',l_arc_dict,'\n')

    '''计算时间余量d(a)'''
    d_arc_dict = {}
    for arc in e_arc_dict.keys():
        d_arc_dict[arc] = l_arc_dict[arc] - e_arc_dict[arc]
    print('d(a)时间余量：\n',d_arc_dict,'\n')

    print('关键路径为：',[x for x in d_arc_dict if d_arc_dict[x] == 0])
# if __name__ == '__main__':
#     vexs = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
#     arcs = [('v1', 'v2', 6), ('v1', 'v3', 4), ('v1', 'v4', 5), ('v2', 'v5', 1), ('v3', 'v5', 1), ('v4', 'v6', 2),
#             ('v5', 'v7', 9), ('v5', 'v8', 7), ('v6', 'v8', 4), ('v7', 'v9', 2), ('v8', 'v9', 4)]
#     G = ALGraph_matrix(vexs, arcs)
#     udc = UDC_matrix()
#     udc.createUDN(G, arcs)
#     ts = TS()
#     vexs = ts.TopologicalSort(G)
#     critical_path(vexs,arcs)


# 希尔排序
# 一次移动，移动位置较大，跳跃式的接近排序后的最终位置
# 最后一次只需要少量移动
# 增量序列必须是递减的，最后一个必须是1
# 增量序列应该是互质的

def shell_sort(nums):
    n = len(nums)
    gap = n//2
    while gap > 0:
        for i in range(gap, n):
            temp = nums[i]
            j = i
            while j >= gap and nums[j-gap] > temp:
                nums[j] = nums[j-gap]
                j -= gap
            nums[j] = temp
        gap = gap // 2
    return nums
# print(shell_sort(nums))

# 简单选择排序
def simple_sort(nums):
    for i in range(len(nums)):
        min_idx = i
        for j in range(i+1,len(nums)):
            if nums[min_idx] > nums[j]:
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
    return nums
# print(simple_sort(nums))
