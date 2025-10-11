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
        #self._entry_point_base = 0
        log_info(f'filename {data.file.filename}')
        # code + const + initc
        self._is_unknown_bin = False
        if 'p0' in data.file.filename:
            self._flash_start = 0x70000000
            self._entry_point = 0x9e5c # from kobjdump

            #self._initc_file_offset = 0x89334 # refer p1, after 69 6C E4 F9 FF 7F 00 00 03 00 00 00
            self._initc_len = 0x220 * 4 # in reset_minim
            self._initc_v_offset = 0x2df8 # in reset_minim
            self._initc_flash_offset = 0x700920c0 #in reset_minim
            self._initc_file_offset = self._initc_flash_offset - self._flash_start
            #700894a4 trapset_bitmap
            self._const_file_offset = 0x891e0 # refer p1, after 16 00 71 48 D8 4C
            self._const_len = self._initc_file_offset - self._const_file_offset
            self._const_flash_offset = self._flash_start + self._const_file_offset
            self._mem_start = 0
        elif 'p1' in data.file.filename:
            self._flash_start = 0x78000000
            self._entry_point = 0x5e2

            self._initc_file_offset = 0x1777c
            self._initc_len = 0x17 * 4
            self._initc_v_offset = 0x43268
            self._initc_flash_offset = 0x7801777c

            self._const_file_offset = 0xcbf6 + 2
            self._const_len = self._initc_file_offset - self._const_file_offset
            self._const_flash_offset = 0x7800cbf8 #8000988a _get_pmalloc_config

            self._mem_start = 0x40000
        else:
            self._is_unknown_bin = True
    @classmethod
    def is_valid_for_data(self, data):
        return True

    def perform_get_address_size(self):
        return KALIMBA.address_size

    def init(self):
        if not self._is_unknown_bin:#If const and initc are uncertain, turn false
            self.add_auto_segment(
                self._entry_point_base + 0x180, self._const_file_offset,# remove 0x180 bytes
                0x180, self._const_file_offset,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable)


            self.add_auto_segment(
                self._const_flash_offset, self.data.length - self._const_file_offset,
                self._const_file_offset, self.data.length - self._const_file_offset, 
                SegmentFlag.SegmentReadable)

            self.add_auto_segment(
                self._initc_v_offset, self._initc_len,
                self._initc_file_offset, self._initc_len, 
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable)
            self.add_entry_point(self._entry_point + self._entry_point_base)
        else:
            '''
            self.add_auto_segment(
                self._entry_point_base + 0x180, self.data.length - 0x180,# remove 0x180 bytes
                0x180, self.data.length - 0x180,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable)
            self.add_entry_point(self._entry_point + self._entry_point_base)
            '''
            self.add_auto_segment(
                0, self.data.length,
                0, self.data.length,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable)
            self.add_entry_point(self._entry_point + self._entry_point_base) # optional
        
        return True

    def perform_is_executable(self):
        return True

    def perform_get_entry_point(self):
        return self._entry_point + self._entry_point_base
