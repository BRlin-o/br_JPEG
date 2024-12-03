from collections import Counter, namedtuple
import pandas as pd
from queue import PriorityQueue
from .huffman_tree_interface import HuffmanTreeInterface

class GuetzliStyleHuffmanTree(HuffmanTreeInterface):
    class _Node:
        def __init__(self, value, freq, left_child, right_child):
            self.value = value
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child

        @staticmethod
        def init_leaf(value, freq):
            return GuetzliStyleHuffmanTree._Node(value, freq, None, None)

        @staticmethod
        def init_node(left_child, right_child):
            freq = left_child.freq + right_child.freq
            return GuetzliStyleHuffmanTree._Node(None, freq, left_child, right_child)

        def is_leaf(self):
            return self.value is not None

        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, data, max_bit_length=16):
        self.max_bit_length = max_bit_length
        self.key_freq = Counter(data)
        self.key_codelen = {}
        self.key_code = {}
        self.__root = self.build_tree()

    def build_tree(self):
        pq = PriorityQueue()
        for symbol, freq in self.key_freq.items():
            pq.put(self._Node.init_leaf(symbol, freq))

        while pq.qsize() > 1:
            left = pq.get()
            right = pq.get()
            parent = self._Node.init_node(left, right)
            pq.put(parent)

        root = pq.get()
        self.assign_code_lengths(root, 0)
        self.adjust_code_lengths()
        return root

    def assign_code_lengths(self, node, depth):
        if node.is_leaf():
            self.key_codelen[node.value] = min(depth, self.max_bit_length)
        else:
            if depth < self.max_bit_length:
                self.assign_code_lengths(node.left_child, depth + 1)
                self.assign_code_lengths(node.right_child, depth + 1)
            else:
                # 當達到最大深度時，將所有子節點視為葉子節點
                self.make_leaves(node, depth)

    def make_leaves(self, node, depth):
        if node.is_leaf():
            self.key_codelen[node.value] = depth
        else:
            self.make_leaves(node.left_child, depth)
            self.make_leaves(node.right_child, depth)

    def adjust_code_lengths(self):
        # 調整代碼長度以滿足 JPEG 標準要求
        while max(self.key_codelen.values()) > self.max_bit_length:
            longest = max(self.key_codelen, key=self.key_codelen.get)
            self.key_codelen[longest] -= 1

    def generate_codes(self):
        # 按照代碼長度和頻率排序符號
        symbols_sorted = sorted(self.key_freq.keys(),
                                key=lambda s: (-self.key_codelen[s], -self.key_freq[s]))
        
        code = 0
        prev_codelen = 0
        for symbol in symbols_sorted:
            codelen = self.key_codelen[symbol]
            if codelen > prev_codelen:
                code <<= (codelen - prev_codelen)
            self.key_code[symbol] = f'{code:0{codelen}b}'
            code += 1
            prev_codelen = codelen

        return self.key_code

    def to_pandas(self):
        data = {
            "symbol": list(self.key_freq.keys()),
            "freq": list(self.key_freq.values()),
            "codelen": [self.key_codelen[symbol] for symbol in self.key_freq],
            "code": [self.key_code.get(symbol, "") for symbol in self.key_freq]
        }
        return pd.DataFrame(data).sort_values(by=['codelen', 'freq'], ascending=[True, False])

    # 其他方法保持不變...