import re
import numpy as np

class ColorComponent:
    def __init__(self, id, hscale, vscale, q_table_index, dc_table, ac_table):
        self.id = id
        self.hscale = hscale
        self.vscale = vscale
        self.q_table_index = q_table_index
        self.dc_table = dc_table
        self.ac_table = ac_table
    
    def __str__(self):
        return f"ColorComponent(id={self.id}, hscale={self.hscale}, vscale={self.vscale}, q_table_index={self.q_table_index}, dc_table={self.dc_table}, ac_table={self.ac_table})"

def binaryCode(x):
    """
    計算二進制編碼的值和長度。
    """
    if x == 0:
        return 0, ""
    vli = binaryCodeLength(abs(x))
    if x > 0:
        return vli, np.binary_repr(x, vli)
    else:  # x < 0
        complement = 2**int(vli) - 1 + x
        return vli, np.binary_repr(complement, vli)

def binaryCodeLength(n):
    """
    計算值的二進制編碼長度。
    """
    if n < 2:
        return 1
    else:
        return np.floor(np.log2(n) + 1).astype(np.uint8)

def EntropyCoding(Bitstream):
    """
    將位流分段並進行熵編碼處理。
    """
    codelen = 8
    perfect_len = (len(Bitstream) + codelen - 1) // codelen
    perfect_code = Bitstream + '1' * (perfect_len * codelen - len(Bitstream))
    split_perfect_code = re.findall(r".{8}", perfect_code)
    ECS = []
    for code in split_perfect_code:
        ECS.append(code)
        if code == "1" * 8:
            ECS.append("0" * 8)
    return ECS

def HuffmanTable2FileStructure(huffmanTable, DEBUG=False):
    """
    將 Huffman 表轉換為 JPEG 文件的格式結構。
    """
    bitsCount = np.zeros((16), dtype=np.uint8)
    sorted_table = dict(sorted(huffmanTable.items(), key=lambda x: (len(x[0]), x[0])))
    if DEBUG:
        print("[DEBUG] sorted_table")
        print(sorted_table)
    codes = [[] for _ in range(16)]
    for code in sorted_table.items():
        if DEBUG:
            print("val={}, code={}, len={}".format(code[1], code[0], len(code[0])))
        bitsCount[len(code[0]) - 1] += 1
        hexCode = int(code[1], 16)
        codes[len(code[0]) - 1].append(hexCode)
    return bitsCount, codes

def convertHuffmanValuesToLists(huffman_table):
    """
    將 Huffman 表的值轉換為列表結構。
    """
    return {key: [value] for key, value in huffman_table.items()}
