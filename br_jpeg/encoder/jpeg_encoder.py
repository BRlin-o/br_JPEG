import os
import io
import struct
import numpy as np
from pathlib import Path
import imageio.v3 as iio

from base_encoder import EncoderBase
from utils.constants import fileDtype, valueDtype
from file.jpeg_file_writer import JPEGFileWriter
from utils.helpers import ColorComponent, binaryCode, EntropyCoding, HuffmanTable2FileStructure, convertHuffmanValuesToLists
from utils.color_transforms import RGB2YCbCr, Gray2YCbCr
from utils.dct import FDCT
from utils.quantization import getQuantizationTable, Quantize
from utils.zigzag import Zigzag, Zigzag1Block
from utils.huffman.standard import StandardHuffmanTree
from utils.huffman.optimized import OptimizedHuffmanTree
from utils.extensions.datahiding import transfor2CATCodeword, transfor2CodewordCAT

class JPEGEncoder(EncoderBase):
    def init(self):
        # 初始化 JPEG 编码器所需的参数
        self.block_size = 8
        self.secret_data = ''
        self.embedded_len = 0

    def __init__(self, image_path=None, quality=None, save_path=None, to_gray=False, huffman_type="std", DEBUG=False):
        self.DEBUG = DEBUG
        if image_path is not None:
            self(image_path, quality, save_path, to_gray, huffman_type)

    def __call__(self, image_path, quality=None, save_path=None, to_gray=False, huffman_type="std"):
        # super().__init__(image_path, DEBUG)
        self.image_path = Path(image_path)
        self.init()
        if quality is not None:
            self.encode(quality, save_path, to_gray, huffman_type)
    
    def load_image(self, image_path):
        self.image = iio.imread(image_path)
        self.image = self.image.astype(np.int32)
        if len(self.image.shape) == 2:  # 灰階影像
            self.image_height, self.image_width = self.image.shape
            num_channels = 1
        else:  # 彩色影像
            self.image_height, self.image_width, num_channels = self.image.shape

        # 計算需要填充的行數和列數
        pad_height = (8 - self.image_height % 8) % 8
        pad_width = (8 - self.image_width % 8) % 8

        # 決定填充參數
        if num_channels == 1:
            padding = ((0, pad_height), (0, pad_width))  # 灰階影像的填充
        else:
            padding = ((0, pad_height), (0, pad_width), (0, 0))  # 彩色影像的填充

        # 使用numpy pad函數進行填充
        if pad_height > 0 or pad_width > 0:
            self.image = np.pad(self.image, padding, mode='edge')

            # 更新影像的高度和寬度
            self.image_height, self.image_width = self.image.shape[:2]

            print(f"Updated image size: {self.image_height}x{self.image_width}")
    
    @staticmethod
    def img2ycbcr(image, to_gray=False):
        print("image.shape", image.shape)
        if image.ndim == 3:
            ycbcr = RGB2YCbCr(image, to_gray=to_gray)
        else:
            ycbcr = Gray2YCbCr(image)
        return ycbcr

    def init_color_components(self, to_gray):
        if to_gray:
            self.color_components = [ColorComponent(id=1, hscale=1, vscale=1, q_table_index=0, dc_table=0, ac_table=0)]
            self.QuantizationTable = [[]]
            self.HuffmanTable = {"DC": [[]], "AC": [[]]}
        else:
            self.color_components = [
                ColorComponent(id=1, hscale=1, vscale=1, q_table_index=0, dc_table=0, ac_table=0),
                ColorComponent(id=2, hscale=1, vscale=1, q_table_index=1, dc_table=1, ac_table=1),
                ColorComponent(id=3, hscale=1, vscale=1, q_table_index=1, dc_table=1, ac_table=1)
            ]
            self.QuantizationTable = [[], []]
            self.HuffmanTable = {"DC": [[], []], "AC": [[], []]}

    def AC_encoding(self, vle_arr, HuffmanACTable):
        """
        对AC系数进行霍夫曼编码。

        :param vle_arr: 变长编码数组。
        :param HuffmanACTable: AC霍夫曼编码表。
        :return: 编码后的字符串。
        """
        ac_encoded = []
        for run, amplitude in vle_arr:
            size, word = binaryCode(amplitude)
            # print("run", run, "\tsize", size, "\tword", word)
            huffman_key = f"{run:01x}{size:01x}"
            # huffman_key = f"{hex(run)[-1]}{hex(size)[-1]}"
            # huffman_key = "{}{}".format(hex(run)[-1], hex(size)[-1])
            # print("huffman_key", huffman_key)
            huffman_codes = HuffmanACTable[huffman_key]
            code_index = 0
            if self.embedded_len < len(self.secret_data) and len(huffman_codes) > 1:
                unit_embed_len = np.log2(len(huffman_codes)).astype(int)
                code_index = int(self.secret_data[self.embedded_len:self.embedded_len+unit_embed_len], 2)
                self.embedded_len+=unit_embed_len
            huffman_code = huffman_codes[code_index]
            if run == 0 and size == 0:
                word = ""
            ac_encoded.append(f"{huffman_code}{word}")
        return ''.join(ac_encoded)

    def HuffmanEncodingMCU(self, mcu):
        """
        对单个MCU进行霍夫曼编码。

        :param mcu: MCU数据，包含DPCM和AC系数。
        :return: 编码后的字符串。
        """
        mcu_encoded = []

        for comp_idx, (dpcm, ac_coefs) in enumerate(mcu):
            comp = self.color_components[comp_idx]

            # DPCM 编码
            code_len, code_word = binaryCode(dpcm)
            huffman_dc_table = self.HuffmanTable["DC"][comp.dc_table]
            huffman_bytes = huffman_dc_table[f"{code_len:02x}"][0]
            mcu_encoded.append(f"{huffman_bytes}{code_word}")

            # AC 编码
            huffman_ac_table = self.HuffmanTable["AC"][comp.ac_table]
            mcu_encoded.append(self.AC_encoding(ac_coefs, huffman_ac_table))

        return ''.join(mcu_encoded)

    def encodingMCUs(self, MCUs):
        """
        对所有MCUs进行霍夫曼编码。

        :param MCUs: 所有MCU的集合。
        :return: 编码后的字符串。
        """
        return ''.join(self.HuffmanEncodingMCU(mcu) for mcu in MCUs)

    @staticmethod
    def getVLE(zigzaged_arr):
        """
        从zigzag扫描的数组中获取VLE（变长编码）。

        :param zigzaged_arr: 经过zigzag扫描的数组。
        :return: VLE数组。
        """
        vle = []
        unZeroIndex = np.argwhere(zigzaged_arr != 0).flatten()
        last_unZero_Index = 0
        for i in unZeroIndex:
            while(i+1-last_unZero_Index > 16):
                # print("{}:{}, len={}".format(last_unZero_Index, last_unZero_Index+16, 16))
                vle.append((15, 0))
                last_unZero_Index = last_unZero_Index+16
            # print("{}:{}, len={}".format(last_unZero_Index, i+1, i+1-last_unZero_Index))
            vle.append((i-last_unZero_Index, zigzaged_arr[i]))
            last_unZero_Index = i+1
        if last_unZero_Index < 63:
            vle.append((0, 0))
        return np.array(vle, dtype=np.int32)
    
    def getMCU(self, zigzaged_arr, showOutput=False):
        """
        从zigzag扫描的数组中获取MCUs。

        :param zigzaged_arr: 经过zigzag扫描的数组。
        :param showOutput: 是否打印输出。
        :return: MCU数组。
        """
        MCUs = []
        last_DC = np.zeros(len(self.color_components))

        for mcu in range(len(zigzaged_arr)):
            newMCU = []  # (DPCM, VLE)
            for comp_idx, comp in enumerate(self.color_components):
                # DPCM
                dpcm = int(zigzaged_arr[mcu][comp_idx][0] - last_DC[comp_idx])
                last_DC[comp_idx] = zigzaged_arr[mcu][comp_idx][0]

                # VLE
                vle = self.getVLE(zigzaged_arr[mcu][comp_idx][1:64])
                newMCU.append((dpcm, vle))

            MCUs.append(newMCU)

        if showOutput:
            print("[ShowOutput] getMCU: len=", len(MCUs))
            for mcu in MCUs:
                print(mcu)

        return MCUs

    def calculateHuffmanFrequencies(self, MCUs, components):
        # 初始化频率统计字典
        dc_frequencies = [{} for _ in range(len(self.HuffmanTable["DC"]))]
        ac_frequencies = [{} for _ in range(len(self.HuffmanTable["AC"]))]

        # 遍历所有MCUs来统计频率
        for mcu in MCUs:
            for comp_index, (dpcm, ac_coefs) in enumerate(mcu):
                # 统计DC系数频率
                dc_key = self.formatHuffmanKey(binaryCode(dpcm)[0])
                self.updateFrequency(dc_frequencies[components[comp_index].dc_table], dc_key)

                # 统计AC系数频率
                for run, amplitude in ac_coefs:
                    ac_key = self.formatHuffmanKey(run, binaryCode(amplitude)[0])
                    self.updateFrequency(ac_frequencies[components[comp_index].ac_table], ac_key)

        return dc_frequencies, ac_frequencies

    def formatHuffmanKey(self, run, size=None):
        """ 格式化霍夫曼键值 """
        if size is not None:
            return "{}{}".format(hex(run)[-1], hex(size)[-1])
        return "{:02x}".format(run)

    def updateFrequency(self, frequency_dict, key):
        """ 更新频率统计 """
        if key in frequency_dict:
            frequency_dict[key] += 1
        else:
            frequency_dict[key] = 1

    def init_huffman_table(self, MCUs, color_components, type="std"):
        if type == "std":
            self.HuffmanTable["DC"][0] = transfor2CATCodeword(StandardHuffmanTree("LUMIN_DC").generate_codes())
            self.HuffmanTable["AC"][0] = transfor2CATCodeword(StandardHuffmanTree("LUMIN_AC").generate_codes())
            if len(color_components) > 1:
                self.HuffmanTable["DC"][1] = transfor2CATCodeword(StandardHuffmanTree("CHROMIN_DC").generate_codes())
                self.HuffmanTable["AC"][1] = transfor2CATCodeword(StandardHuffmanTree("CHROMIN_AC").generate_codes())
        elif type == "opt":
            DC_freq, AC_freq = self.calculateHuffmanFrequencies(MCUs, color_components)
            opt_DC_huffman_tree = OptimizedHuffmanTree(DC_freq[0], 16)
            self.HuffmanTable["DC"][0] = convertHuffmanValuesToLists(opt_DC_huffman_tree.generate_codes())
            opt_AC_huffman_tree = OptimizedHuffmanTree(AC_freq[0], 16)
            self.HuffmanTable["AC"][0] = convertHuffmanValuesToLists(opt_AC_huffman_tree.generate_codes())
            if len(color_components) > 1:
                opt_DC_huffman_tree = OptimizedHuffmanTree(DC_freq[1], 16)
                self.HuffmanTable["DC"][1] = convertHuffmanValuesToLists(opt_DC_huffman_tree.generate_codes())
                opt_AC_huffman_tree = OptimizedHuffmanTree(AC_freq[1], 16)
                self.HuffmanTable["AC"][1] = convertHuffmanValuesToLists(opt_AC_huffman_tree.generate_codes())

    ## huffman_type=["std", "opt", "auto"]
    def encode(self, quality, save_path=None, to_gray=False, huffman_type="std"):
        self.init_color_components(to_gray)
        self.quality = quality
        self.huffman_type = huffman_type

        # 加载和处理图像
        self.load_image(self.image_path)

        # 颜色空间转换、DCT、量化、霍夫曼编码等
        self.ycbcr = self.img2ycbcr(self.image, to_gray)
        
        self.fdct = FDCT(self.ycbcr, self.block_size)
        
        q_table = getQuantizationTable(quality)
        if len(self.color_components) > 1:
            self.QuantizationTable = q_table
        else:
            self.QuantizationTable = [q_table[0]]
        self.quantized = np.zeros(self.ycbcr.shape, dtype=fileDtype)
        for i in range(len(self.color_components)):
            self.quantized[:, :, i] = Quantize(self.fdct[:, :, i], self.QuantizationTable[self.color_components[i].q_table_index])
        
        self.zigzaged = Zigzag(self.quantized, self.block_size)
        
        # # Minimum Coded Unit
        self.MCUs = self.getMCU(self.zigzaged)
        
        self.init_huffman_table(self.MCUs, self.color_components, self.huffman_type)

        self.encodedBitstream = self.encodingMCUs(self.MCUs)

        # # 将编码后的数据写入文件
        self.save_path = self.save_encoded_image(save_path)
        if save_path is None:
            print(f"文件已保存至 {self.save_path}")

    def save_encoded_image(self, encodedBitstream=None, save_path=None):
        if encodedBitstream is None:
            encodedBitstream = self.encodedBitstream
        # 如果没有指定保存路径，生成默认的保存路径和文件名
        if save_path is None:
            original_filename_stem = self.image_path.stem  # 原始文件名（不含扩展名）
            grayscale_suffix = "_G" if len(self.color_components) == 1 else ""
            quality_suffix = f"_Q{self.quality}"
            huffman_suffix = "_opt" if self.huffman_type == "opt" else ""
            default_filename = f"{original_filename_stem}{grayscale_suffix}{quality_suffix}{huffman_suffix}.jpg"
            save_path = JPEGFileWriter.DEFAULT_FOLDER / default_filename

        huffman_tables = {"DC": [], "AC": []}
        for idx, table_type in enumerate(["DC", "AC"]):
            for table_id in range(len(self.HuffmanTable[table_type])):
                huffman_tables[table_type].append(transfor2CodewordCAT(self.HuffmanTable[table_type][table_id]))

        # print("self.HuffmanTable", self.HuffmanTable)
        # print("huffman_tables", huffman_tables)

        # 创建JPEG文件写入器
        writer = JPEGFileWriter(
            image_height=self.image_height,
            image_width=self.image_width,
            block_size=self.block_size,
            quantization_tables=self.QuantizationTable,
            huffman_tables=huffman_tables,
            color_components=self.color_components
        )

        # 写入文件
        self.data = writer.write_file(encodedBitstream, save_path)
        return save_path

if __name__ == "__main__":
    # Method 1
    encoder = JPEGEncoder("./datasets/baboon.tiff")
    encoder.encode(quality=30, to_gray=True, huffman_type="opt")

    # Method 2
    encoder = JPEGEncoder()
    encoder("./datasets/baboon.tiff", quality=30, to_gray=True, huffman_type="std")

    # # Method 3
    # JPEGEncoder("./datasets/baboon.tiff", quality=30, to_gray=True, huffman_type="opt")