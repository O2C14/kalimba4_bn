import re

def _xuv_read(filename):
    """Read an XUV file
       filename: The path and filename of the xuv file.
       returns:  A dict containing address as key and values as value."""
    p = re.compile('\s*@(?P<addr>[0-9a-fA-F]+) +(?P<value>[0-9a-fA-F]+)\s*')
    values = {}
    with open(filename, 'r') as f:
        for line in f:
            m = p.match(line)
            if m:
                addr = int(m.group('addr'), 16)
                value = int(m.group('value'), 16)
                values[addr] = value
    return values


def _xuv2bin(target, source):
    """Convert an XUV file into binary (uint16 little endian)"""
    values = _xuv_read(source)
    with open(target, 'wb') as f:
        for i in range(min(values), max(values)+1):
            t = values.get(i, 0xFFFF)
            f.write(bytes([t & 0xFF]))
            f.write(bytes([t >> 8]))

_xuv2bin(
    
    r'flash_image.xuv_apps_p1.bin',
    r'e:\Download\qcc_all\qcc-hci\qcc_hci\workspace\QCC3095\image\20251009175228\output\flash_image.xuv_apps_p1.xuv')