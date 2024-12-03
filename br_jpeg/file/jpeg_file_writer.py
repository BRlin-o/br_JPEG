import os
import io
import struct
import numpy as np
from pathlib import Path
from br_jpeg.utils.constants import fileDtype, valueDtype
from br_jpeg.utils.bitstream import EntropyCoding, HuffmanTable2FileStructure
from br_jpeg.utils.zigzag import Zigzag1Block

class JPEGFileWriter:
    DEFAULT_FOLDER = Path("./output/")
    def __init__(self, image_height, image_width, block_size, quantization_tables, huffman_tables, color_components):
        self.image_height, self.image_width = image_height, image_width
        self.quantization_tables = quantization_tables
        self.huffman_tables = huffman_tables
        self.color_components = color_components
        self.block_size = block_size

    def write_soi(self):
        soi = io.BytesIO()
        soi.write(b'\xFF\xD8')
        return soi.getvalue()

    def write_app0(self):
        app0 = io.BytesIO()
        app0.write(b'\xFF\xE0') ## Marker
        app0.write(struct.pack('>H', 16)) ## Length
        app0.write(struct.pack('5s', b"JFIF\0")) ## identifier
        app0.write(struct.pack('>B', 1)) ## JFIF version 1
        app0.write(struct.pack('>B', 1)) ## JFIF version .1
        app0.write(struct.pack('>B', 1)) ## units
        app0.write(struct.pack('>H', 96)) ## x-density
        app0.write(struct.pack('>H', 96)) ## y-density
        app0.write(struct.pack('>B', 0)) ## x-thumbnail
        app0.write(struct.pack('>B', 0)) ## y-thumbnail
        return app0.getvalue()

    def write_dqt(self, q_table, table_id):
        _q_table = q_table.astype(fileDtype)
        dqt = io.BytesIO()
        dqt.write(b'\xFF\xDB') ## Marker
        dqt.write(struct.pack('>H', 2+1+len(_q_table.flatten()))) ## Length((2))
        dqt.write(struct.pack('>B', table_id)) ## 0: luminance((1))
        for quantization in Zigzag1Block(block=_q_table, block_size=self.block_size): ## ((64))
            dqt.write(struct.pack('>B', quantization))
        return dqt.getvalue()

    def write_sof(self):
        sof = io.BytesIO()
        sof = io.BytesIO()
        sof.write(b'\xFF\xC0') ## Marker
        sof.write(struct.pack('>H', 2+1+2+2+1+len(self.color_components)*3)) ## Length
        sof.write(struct.pack('>B', 8)) ## 8: precision
        sof.write(struct.pack('>H', self.image_height)) ## height
        sof.write(struct.pack('>H', self.image_width)) ## width
        sof.write(struct.pack('>B', len(self.color_components))) ## component count
        for color_comp in self.color_components:
            sof.write(struct.pack('>B', color_comp.id))
            sof.write(struct.pack('>B', color_comp.hscale*0x10 + color_comp.vscale*0x01))
            sof.write(struct.pack('>B', color_comp.q_table_index))
        return sof.getvalue()

    def write_dht(self, huffman_table, table_id, is_ac):
        bitsCount, codes = HuffmanTable2FileStructure(huffman_table)
        dht = io.BytesIO()
        dht.write(b'\xFF\xC4')
        dht.write(struct.pack('>H', 2+1+16+bitsCount.sum().astype(np.uint16)))
        dht.write(struct.pack('>B', is_ac*0x10 + table_id*0x01))
        dht.write(struct.pack('16B', *bitsCount))
        for len_codes in codes:
            for code in len_codes:
                dht.write(struct.pack('>B', code))
        return dht.getvalue()

    def write_sos(self, bitstream):
        ecs_list = EntropyCoding(bitstream)
        sos = io.BytesIO()
        sos.write(b'\xFF\xDA')
        sos.write(struct.pack('>H', 2+1+len(self.color_components)*2+1+2))
        sos.write(struct.pack('>B', len(self.color_components)))
        for num_comp in self.color_components:
            sos.write(struct.pack('>B', num_comp.id))
            sos.write(struct.pack('>B', num_comp.dc_table*0x10 + num_comp.ac_table*0x01))
        sos.write(struct.pack('>H', 63)) ## spectral select
        sos.write(struct.pack('>B', 0)) ## successive approx.
        for ecs_block in ecs_list:
            sos.write(struct.pack('>B', int(ecs_block, 2)))
        return sos.getvalue()

    def write_eoi(self):
        eoi = io.BytesIO()
        eoi.write(b'\xFF\xD9')
        return eoi.getvalue()

    def write_file(self, bitstream, save_path):
        with io.BytesIO() as data:
            data.write(self.write_soi())
            data.write(self.write_app0())
            unique_q_table_indices = list({comp.q_table_index for comp in self.color_components})
            for q_table_index in unique_q_table_indices:
                q_table = self.quantization_tables[q_table_index]
                data.write(self.write_dqt(q_table, q_table_index))
            data.write(self.write_sof())
            huffman_table_entries = [
                (table_type, table_id)
                for table_type in ["DC", "AC"]
                for table_id in range(len(self.huffman_tables[table_type]))
            ]
            for table_type, table_id in huffman_table_entries:
                data.write(self.write_dht(self.huffman_tables[table_type][table_id], table_id, table_type == "AC"))
            data.write(self.write_sos(bitstream))
            data.write(self.write_eoi())

            self.ensure_directory_exists(save_path)
            with open(save_path, "wb") as f:
                f.write(data.getvalue())
            return data.getvalue()

    def create_marker_segment(marker, data):
        segment = io.BytesIO()
        segment.write(marker)
        for item in data:
            if isinstance(item, tuple):  # 如果是元组，表示需要打包
                segment.write(struct.pack(*item))
            else:
                segment.write(item)
        return segment

    def write_soi(self):
        # 返回 SOI 段的字节
        return b'\xFF\xD8'

    def ensure_directory_exists(self, file_path):
        os.makedirs(Path(file_path).parent, exist_ok=True)