from .huffman_tree_interface import HuffmanTreeInterface
from collections import Counter, namedtuple

class StandardHuffmanTree(HuffmanTreeInterface):
    """
    StandardHuffmanTree 用於存儲和檢索標準霍夫曼表。
    它將霍夫曼表設計為靜態成員，讓所有 StandardHuffmanTree 實例共享同一組霍夫曼表數據，
    從而節省記憶體空間並保持代碼的整潔和模組化。

    Attributes:
        table_type (str): 霍夫曼表的類型（如 'LUMIN_DC', 'CHROMIN_DC', 等）。
    """
    HuffmanTables = namedtuple('HuffmanTables', ['LUMIN_DC', 'CHROMIN_DC', 'LUMIN_AC', 'CHROMIN_AC'])
    standard_huffman_tables = HuffmanTables(
        LUMIN_DC={
            '00': '00', 
            '010': '01', 
            '011': '02', 
            '100': '03', 
            '101': '04', 
            '110': '05', 
            '1110': '06', 
            '11110': '07', 
            '111110': '08', 
            '1111110': '09', 
            '11111110': '0a', 
            '111111110': '0b'
        }, 

        CHROMIN_DC={
            '00': '00', 
            '01': '01', 
            '10': '02', 
            '110': '03', 
            '1110': '04', 
            '11110': '05', 
            '111110': '06', 
            '1111110': '07', 
            '11111110': '08', 
            '111111110': '09', 
            '1111111110': '0a', 
            '11111111110': '0b'
        }, 

        LUMIN_AC={
            '1010': '00', 
            '00': '01', 
            '01': '02', 
            '100': '03', 
            '1011': '04', 
            '11010': '05', 
            '1111000': '06', 
            '11111000': '07', 
            '1111110110': '08', 
            '1111111110000010': '09', 
            '1111111110000011': '0a', 
            '1100': '11', 
            '11011': '12', 
            '1111001': '13', 
            '111110110': '14', 
            '11111110110': '15', 
            '1111111110000100': '16', 
            '1111111110000101': '17', 
            '1111111110000110': '18', 
            '1111111110000111': '19', 
            '1111111110001000': '1a', 
            '11100': '21', 
            '11111001': '22', 
            '1111110111': '23', 
            '111111110100': '24', 
            '1111111110001001': '25', 
            '1111111110001010': '26', 
            '1111111110001011': '27', 
            '1111111110001100': '28', 
            '1111111110001101': '29', 
            '1111111110001110': '2a', 
            '111010': '31', 
            '111110111': '32', 
            '111111110101': '33', 
            '1111111110001111': '34', 
            '1111111110010000': '35', 
            '1111111110010001': '36', 
            '1111111110010010': '37', 
            '1111111110010011': '38', 
            '1111111110010100': '39', 
            '1111111110010101': '3a', 
            '111011': '41', 
            '1111111000': '42', 
            '1111111110010110': '43', 
            '1111111110010111': '44', 
            '1111111110011000': '45', 
            '1111111110011001': '46', 
            '1111111110011010': '47', 
            '1111111110011011': '48', 
            '1111111110011100': '49', 
            '1111111110011101': '4a', 
            '1111010': '51', 
            '11111110111': '52', 
            '1111111110011110': '53', 
            '1111111110011111': '54', 
            '1111111110100000': '55', 
            '1111111110100001': '56', 
            '1111111110100010': '57', 
            '1111111110100011': '58', 
            '1111111110100100': '59', 
            '1111111110100101': '5a', 
            '1111011': '61', 
            '111111110110': '62', 
            '1111111110100110': '63', 
            '1111111110100111': '64', 
            '1111111110101000': '65', 
            '1111111110101001': '66', 
            '1111111110101010': '67', 
            '1111111110101011': '68', 
            '1111111110101100': '69', 
            '1111111110101101': '6a', 
            '11111010': '71', 
            '111111110111': '72', 
            '1111111110101110': '73', 
            '1111111110101111': '74', 
            '1111111110110000': '75', 
            '1111111110110001': '76', 
            '1111111110110010': '77', 
            '1111111110110011': '78', 
            '1111111110110100': '79', 
            '1111111110110101': '7a', 
            '111111000': '81', 
            '111111111000000': '82', 
            '1111111110110110': '83', 
            '1111111110110111': '84', 
            '1111111110111000': '85', 
            '1111111110111001': '86', 
            '1111111110111010': '87', 
            '1111111110111011': '88', 
            '1111111110111100': '89', 
            '1111111110111101': '8a', 
            '111111001': '91', 
            '1111111110111110': '92', 
            '1111111110111111': '93', 
            '1111111111000000': '94', 
            '1111111111000001': '95', 
            '1111111111000010': '96', 
            '1111111111000011': '97', 
            '1111111111000100': '98', 
            '1111111111000101': '99', 
            '1111111111000110': '9a', 
            '111111010': 'a1', 
            '1111111111000111': 'a2', 
            '1111111111001000': 'a3', 
            '1111111111001001': 'a4', 
            '1111111111001010': 'a5', 
            '1111111111001011': 'a6', 
            '1111111111001100': 'a7', 
            '1111111111001101': 'a8', 
            '1111111111001110': 'a9', 
            '1111111111001111': 'aa', 
            '1111111001': 'b1', 
            '1111111111010000': 'b2', 
            '1111111111010001': 'b3', 
            '1111111111010010': 'b4', 
            '1111111111010011': 'b5', 
            '1111111111010100': 'b6', 
            '1111111111010101': 'b7', 
            '1111111111010110': 'b8', 
            '1111111111010111': 'b9', 
            '1111111111011000': 'ba', 
            '1111111010': 'c1', 
            '1111111111011001': 'c2', 
            '1111111111011010': 'c3', 
            '1111111111011011': 'c4', 
            '1111111111011100': 'c5', 
            '1111111111011101': 'c6', 
            '1111111111011110': 'c7', 
            '1111111111011111': 'c8', 
            '1111111111100000': 'c9', 
            '1111111111100001': 'ca', 
            '11111111000': 'd1', 
            '1111111111100010': 'd2', 
            '1111111111100011': 'd3', 
            '1111111111100100': 'd4', 
            '1111111111100101': 'd5', 
            '1111111111100110': 'd6', 
            '1111111111100111': 'd7', 
            '1111111111101000': 'd8', 
            '1111111111101001': 'd9', 
            '1111111111101010': 'da', 
            '1111111111101011': 'e1', 
            '1111111111101100': 'e2', 
            '1111111111101101': 'e3', 
            '1111111111101110': 'e4', 
            '1111111111101111': 'e5', 
            '1111111111110000': 'e6', 
            '1111111111110001': 'e7', 
            '1111111111110010': 'e8', 
            '1111111111110011': 'e9', 
            '1111111111110100': 'ea', 
            '11111111001': 'f0', 
            '1111111111110101': 'f1', 
            '1111111111110110': 'f2', 
            '1111111111110111': 'f3', 
            '1111111111111000': 'f4', 
            '1111111111111001': 'f5', 
            '1111111111111010': 'f6', 
            '1111111111111011': 'f7', 
            '1111111111111100': 'f8', 
            '1111111111111101': 'f9', 
            '1111111111111110': 'fa'
        }, 

        CHROMIN_AC={'00': '00', 
            '01': '01', 
            '100': '02', 
            '1010': '03', 
            '11000': '04', 
            '11001': '05', 
            '111000': '06', 
            '1111000': '07', 
            '111110100': '08', 
            '1111110110': '09', 
            '111111110100': '0a', 
            '1011': '11', 
            '111001': '12', 
            '11110110': '13', 
            '111110101': '14', 
            '11111110110': '15', 
            '111111110101': '16', 
            '1111111110001000': '17', 
            '1111111110001001': '18', 
            '1111111110001010': '19', 
            '1111111110001011': '1a', 
            '11010': '21', 
            '11110111': '22', 
            '1111110111': '23', 
            '111111110110': '24', 
            '111111111000010': '25', 
            '1111111110001100': '26', 
            '1111111110001101': '27', 
            '1111111110001110': '28', 
            '1111111110001111': '29', 
            '1111111110010000': '2a', 
            '11011': '31', 
            '11111000': '32', 
            '1111111000': '33', 
            '111111110111': '34', 
            '1111111110010001': '35', 
            '1111111110010010': '36', 
            '1111111110010011': '37', 
            '1111111110010100': '38', 
            '1111111110010101': '39', 
            '1111111110010110': '3a', 
            '111010': '41', 
            '111110110': '42', 
            '1111111110010111': '43', 
            '1111111110011000': '44', 
            '1111111110011001': '45', 
            '1111111110011010': '46', 
            '1111111110011011': '47', 
            '1111111110011100': '48', 
            '1111111110011101': '49', 
            '1111111110011110': '4a', 
            '111011': '51', 
            '1111111001': '52', 
            '1111111110011111': '53', 
            '1111111110100000': '54', 
            '1111111110100001': '55', 
            '1111111110100010': '56', 
            '1111111110100011': '57', 
            '1111111110100100': '58', 
            '1111111110100101': '59', 
            '1111111110100110': '5a', 
            '1111001': '61', 
            '11111110111': '62', 
            '1111111110100111': '63', 
            '1111111110101000': '64', 
            '1111111110101001': '65', 
            '1111111110101010': '66', 
            '1111111110101011': '67', 
            '1111111110101100': '68', 
            '1111111110101101': '69', 
            '1111111110101110': '6a', 
            '1111010': '71', 
            '11111111000': '72', 
            '1111111110101111': '73', 
            '1111111110110000': '74', 
            '1111111110110001': '75', 
            '1111111110110010': '76', 
            '1111111110110011': '77', 
            '1111111110110100': '78', 
            '1111111110110101': '79', 
            '1111111110110110': '7a', 
            '11111001': '81', 
            '1111111110110111': '82', 
            '1111111110111000': '83', 
            '1111111110111001': '84', 
            '1111111110111010': '85', 
            '1111111110111011': '86', 
            '1111111110111100': '87', 
            '1111111110111101': '88', 
            '1111111110111110': '89', 
            '1111111110111111': '8a', 
            '111110111': '91', 
            '1111111111000000': '92', 
            '1111111111000001': '93', 
            '1111111111000010': '94', 
            '1111111111000011': '95', 
            '1111111111000100': '96', 
            '1111111111000101': '97', 
            '1111111111000110': '98', 
            '1111111111000111': '99', 
            '1111111111001000': '9a', 
            '111111000': 'a1', 
            '1111111111001001': 'a2', 
            '1111111111001010': 'a3', 
            '1111111111001011': 'a4', 
            '1111111111001100': 'a5', 
            '1111111111001101': 'a6', 
            '1111111111001110': 'a7', 
            '1111111111001111': 'a8', 
            '1111111111010000': 'a9', 
            '1111111111010001': 'aa', 
            '111111001': 'b1', 
            '1111111111010010': 'b2', 
            '1111111111010011': 'b3', 
            '1111111111010100': 'b4', 
            '1111111111010101': 'b5', 
            '1111111111010110': 'b6', 
            '1111111111010111': 'b7', 
            '1111111111011000': 'b8', 
            '1111111111011001': 'b9', 
            '1111111111011010': 'ba', 
            '111111010': 'c1', 
            '1111111111011011': 'c2', 
            '1111111111011100': 'c3', 
            '1111111111011101': 'c4', 
            '1111111111011110': 'c5', 
            '1111111111011111': 'c6', 
            '1111111111100000': 'c7', 
            '1111111111100001': 'c8', 
            '1111111111100010': 'c9', 
            '1111111111100011': 'ca', 
            '11111111001': 'd1', 
            '1111111111100100': 'd2', 
            '1111111111100101': 'd3', 
            '1111111111100110': 'd4', 
            '1111111111100111': 'd5', 
            '1111111111101000': 'd6', 
            '1111111111101001': 'd7', 
            '1111111111101010': 'd8', 
            '1111111111101011': 'd9', 
            '1111111111101100': 'da', 
            '11111111100000': 'e1', 
            '1111111111101101': 'e2', 
            '1111111111101110': 'e3', 
            '1111111111101111': 'e4', 
            '1111111111110000': 'e5', 
            '1111111111110001': 'e6', 
            '1111111111110010': 'e7', 
            '1111111111110011': 'e8', 
            '1111111111110100': 'e9', 
            '1111111111110101': 'ea', 
            '1111111010': 'f0', 
            '111111111000011': 'f1', 
            '1111111111110110': 'f2', 
            '1111111111110111': 'f3', 
            '1111111111111000': 'f4', 
            '1111111111111001': 'f5', 
            '1111111111111010': 'f6', 
            '1111111111111011': 'f7', 
            '1111111111111100': 'f8', 
            '1111111111111101': 'f9', 
            '1111111111111110': 'fa'
        }
    )
    def __init__(self, table_type):
        if table_type not in ['LUMIN_DC', 'CHROMIN_DC', 'LUMIN_AC', 'CHROMIN_AC']:
            raise ValueError("Invalid Huffman table type")
        self.table_type = table_type

    def build_tree(self, data):
        # 對於標準霍夫曼樹，此方法不執行任何操作
        pass

    def generate_codes(self):
        # 返回指定類型的霍夫曼編碼表
        return getattr(self.standard_huffman_tables, self.table_type)

    def calculate_compressed_size(self, data):
        """
        计算使用标准霍夫曼编码压缩数据后的大小。

        Args:
            data (str/list): 要进行霍夫曼编码的数据，可以是字符串或字符列表。

        Returns:
            int: 使用霍夫曼编码压缩后的数据大小（以比特为单位）。
        """
        if not isinstance(data, (str, list)):
            raise TypeError("Data must be a string or a list of characters")

        freq_dict = Counter(data)
        huffman_table = self.generate_codes()
        compressed_size = 0

        for char, freq in freq_dict.items():
            if char in huffman_table:
                compressed_size += len(huffman_table[char]) * freq
            else:
                raise ValueError(f"Character '{char}' not found in Huffman table")

        return compressed_size

    def to_dict(self):
        # 將指定類型的霍夫曼編碼表轉換為字典
        return dict(getattr(self.standard_huffman_tables, self.table_type))
