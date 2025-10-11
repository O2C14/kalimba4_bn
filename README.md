# 如何使用:
## 对于p0固件
1. 使用xuv2bin.py转换xuv文件并使用kobjdump反汇编前几个字节并修改`self._entry_point`
2. 使用`self._is_unknown_bin = True`来加载完整的固件
3. 分析`reset_minim`也就是entry_point中的第三个循环以获取initc节的信息
4. 搜索`16 00 71 48 D8 4C`, const节在这之后
5. 取消`self._is_unknown_bin = True`并重新加载固件

注意: 前0x180字节会被忽略
# How to use:
## For p0
1. Use xuv2bin.py to convert the xuv file, then disassemble the first few bytes with kobjdump and modify `self._entry_point`
2. Use `self._is_unknown_bin = True` to load the full firmware
3. Analyze `reset_minim` (the third loop in entry_point) to obtain initc section information
4. Search for `16 00 71 48 D8 4C`; the const section follows this address
5. Remove `self._is_unknown_bin = True` and reload the firmware

Note: The first 0x180 bytes will be ignored