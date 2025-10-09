from binaryninja.binaryview import BinaryView
from binaryninja.enums import SegmentFlag
from binaryninja.architecture import Architecture
from binaryninja.log import log_info
from .k_arch import KALIMBA
class KALIMBAView(BinaryView):
    name = 'KALIMBAView'
    long_name = 'KALIMBAView ROM'

    def __init__(self, data):
        BinaryView.__init__(self, parent_view = data, file_metadata = data.file)
        # self.platform = Architecture['KALIMBA'].standalone_platform
        self.data = data
        self.platform = Architecture[KALIMBA.name].standalone_platform
        self.arch = Architecture[KALIMBA.name]
        self._entry_point_base = 0x80000000
        log_info(f'filename {data.file.filename}')
        # code + const + initc
        if 'p0' in data.file.filename:
            self._entry_point = 0x9e5c # from kobjdump

            self._initc_file_offset = 0x89334 # refer p1, after 69 6C E4 F9 FF 7F 00 00 03 00 00 00
            self._initc_len = 0x220 * 4 # in reset_minim
            self._initc_v_offset = 0x2df8 # in reset_minim
            self._initc_flash_offset = 0x700920c0 #in reset_minim

            self._const_file_offset = 0x89062 # refer p1, after _trapset_bitmap_length
            self._const_len = self._initc_file_offset - self._const_file_offset
            self._const_flash_offset = self._initc_flash_offset - self._initc_len - self._const_len
            self._mem_start = 0x70000000
        else:
            self._entry_point = 0x5e2

            self._initc_file_offset = 0x175fc
            self._initc_len = 0x17 * 4
            self._initc_v_offset = 0x43268
            self._initc_flash_offset = 0x7801777c

            self._const_file_offset = 0xca76 + 2
            self._const_len = self._initc_file_offset - self._const_file_offset
            self._const_flash_offset = self._initc_flash_offset - self._initc_len - self._const_len

            self._mem_start = 0x78000000
    @classmethod
    def is_valid_for_data(self, data):
        return True

    def perform_get_address_size(self):
        return KALIMBA.address_size

    def init(self):
        if True:
            self.add_auto_segment(
                self._entry_point_base + 0x180, self._const_file_offset,# remove 0x180 bytes
                0, self._const_file_offset,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable)


            self.add_auto_segment(
                self._const_flash_offset, self.data.length - self._const_file_offset,
                self._const_file_offset, self.data.length - self._const_file_offset, 
                SegmentFlag.SegmentReadable)

            self.add_auto_segment(
                self._initc_v_offset, self.data.length - self._initc_file_offset,
                self._initc_file_offset, self.data.length - self._initc_file_offset, 
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable)

        self.add_entry_point(self._entry_point + self._entry_point_base)
        
        return True

    def perform_is_executable(self):
        return True

    def perform_get_entry_point(self):
        return self._entry_point+self._entry_point_base
#1b9ac _init_pmalloc