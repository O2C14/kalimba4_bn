from struct import unpack
from k_instr_new import *

cond_3bit_map_to_4bit = [0, 1, 3, 10, 11, 12, 13, 15]

pushm_reg_bitmap = [
    0x0000, 0x0002, 0x0006, 0x000e, 0x001e, 0x003e, 0x007e, 0x00fe,
    0x0100, 0x0004, 0x000c, 0x001c, 0x003c, 0x007c, 0x00fc, 0x01fc,
    0x0200, 0x0008, 0x0018, 0x0038, 0x0078, 0x00f8, 0x01f8, 0x03f8,
    0x0400, 0x0010, 0x0030, 0x0070, 0x00f0, 0x01f0, 0x03f0, 0x07f0,
    0x0800, 0x0020, 0x0060, 0x00e0, 0x01e0, 0x03e0, 0x07e0, 0x0fe0,
    0x1000, 0x0040, 0x00c0, 0x01c0, 0x03c0, 0x07c0, 0x0fc0, 0x1fc0,
    0x8000, 0x0080, 0x0180, 0x0380, 0x0780, 0x0f80, 0x1f80, 0x3f80]
# no XOR?
k5_logical_immediate = [
    0x1, 0x2, 0x3, 0x4, 0x7, 0x8, 0xF, 0x10,
    0x1F, 0x20, 0x3F, 0x40, 0x7F, 0x80, 0xFF, 0x100,
    0x1FF, 0x200, 0x3FF, 0x400, 0x7FF, 0x800, 0xFFF, 0x1000,
    0x7FFF, 0x8000, 0xFFFF, 0x10000, 0xFFFFF, 0x7FF0000, 0x7FFFFFF, 0x80000000
]
k5_shift_immediate = [
    -1, -2, -3, -4, -5, -6, -7, -8,
    -12, -14, -16, -20, -22, -24, -31, -32,
    1, 2, 3, 4, 5, 6, 7, 8,
    12, 14, 16, 20, 22, 24, 31, 32,
]


def nbits_unsigned_to_signed(x, bits):
    if (x & (1 << (bits - 1))) != 0:
        return - (((~x) & ((1 << bits) - 1)) + 1)
    else:
        return x


def kalimba_minim_decode_regs_a(instruction: int, banka=KalimbaBank1Reg, bankb=KalimbaBank1Reg, bankc=KalimbaBank1Reg):
    rega = banka(get_bits(instruction, 3, 3))
    regb = bankb(get_bits(instruction, 6, 3))
    regc = bankc(get_bits(instruction, 0, 3))

    return (rega, regb, regc)


def kalimba_minim_decode_a_op_b_typea(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    return KalimbaBinOp(op, regc, rega, regb)


def kalimba_minim_decode_a_op_b_typeb(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    K = get_bits(instruction, 6, 5)
    if op in [KalimbaOp.AND, KalimbaOp.OR]:
        K = k5_logical_immediate[K]
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT]:
        K = k5_shift_immediate[K]
    return KalimbaBinOp(op, regc, rega, K)


def kalimba_minim_decode_se8_16(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    return KalimbaUnOp(op, regc, rega)


def kalimba_minim_decode_mov_add_a(instruction, op):
    a = (get_bits(instruction, 6, 2) << 3) + get_bits(instruction, 3, 3)
    c = (get_bits(instruction, 8, 2) << 3) + get_bits(instruction, 0, 3)
    if a < 16:
        rega = KalimbaBank1Reg(a)
    else:
        rega = KalimbaBank2Reg(a - 16)
    if c < 16:
        regc = KalimbaBank1Reg(c)
    else:
        regc = KalimbaBank2Reg(c - 16)
    if get_bits(instruction, 10, 1) == 0:
        regb = KalimbaBank1Reg.Null
    else:
        regb = regc
    return KalimbaBinOp(op, regc, rega, regb)


def kalimba_minim_decode_mov_add_b(instruction, op):
    c = (get_bits(instruction, 8, 2) << 3) + get_bits(instruction, 0, 3)
    if c < 16:
        regc = KalimbaBank1Reg(c)
    else:
        regc = KalimbaBank2Reg(c - 16)
    if get_bits(instruction, 10, 1) == 0:
        regb = KalimbaBank1Reg.Null
    else:
        regb = regc
    K = (get_bits(instruction, 11, 1) << 5) + get_bits(instruction, 3, 5)
    K = nbits_unsigned_to_signed(K, 6)
    return KalimbaBinOp(op, regc, regb, K)  # TODO K


def kalimba_minim_decode_add_sub_b(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    K = (get_bits(instruction, 11, 2) << 4) + get_bits(instruction, 6, 4)
    return KalimbaBinOp(op, regc, regb, K)


def kalimba_minim_decode_fp_adjust_b(instruction, op):
    regc = KalimbaBank1Reg(get_bits(instruction, 0, 3))
    K = (get_bits(instruction, 14, 1) << 6) + get_bits(instruction, 6, 6)
    return KalimbaBinOp(op, regc, KalimbaBank3Reg.FP, K)


def kalimba_minim_decode_pushm_popm(instruction, op):
    reg_bitmap = pushm_reg_bitmap[get_bits(instruction, 0, 6) - 8]
    FP = get_bits(instruction, 6, 1)
    rLink = get_bits(instruction, 7, 1)
    adj = get_bits(instruction, 8, 2) * 16
    reg_list = []
    if FP == 1:
        reg_list.append(KalimbaBank3Reg.FP)
    for i in range(16):
        if ((reg_bitmap >> i) & 1) != 0:
            reg_list.append(KalimbaBank1Reg(i))
    if rLink == 1:
        reg_list.append(KalimbaBank1Reg.rLink)
    return KalimbaStackOp(op, reg_list, KalimbaCond.Always, adj=adj)


def kalimba_minim_decode_call_jump_regc(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    return KalimbaControlFlow(op, regc, KalimbaCond.Always, None)


def kalimba_minim_decode_call_jump_do_k(instruction, op):
    # TODO K
    cond = KalimbaCond.Always
    if op == KalimbaOp.CALL:
        K = nbits_unsigned_to_signed(instruction & 0xff, 9)
    elif op == KalimbaOp.JUMP:
        K = nbits_unsigned_to_signed(instruction & 0xff, 9)
        cond = KalimbaCond(cond_3bit_map_to_4bit[get_bits(instruction, 9, 3)])
    elif op == KalimbaOp.DOLOOP:
        K = instruction & 0x3f
    return KalimbaControlFlow(op, K * 2, cond, None)


def kalimba_minim_decode_sp_adjust(instruction, op):
    K = nbits_unsigned_to_signed(get_bits(instruction, 0, 6), 6)
    return KalimbaBinOp(op, KalimbaBank3Reg.SP, KalimbaBank3Reg.SP, K)


def kalimba_minim_decode_div(instruction, op):
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    return KalimbaBinOp(op, KalimbaBank3Reg.DivResult, regc, rega)


def kalimba_minim_decode_div_result(instruction, op):
    regc = KalimbaBank1Reg(get_bits(instruction, 0, 3))
    if get_bits(instruction, 3, 1) == 1:
        rega = KalimbaBank3Reg.DivRemainder
    else:
        rega = KalimbaBank3Reg.DivResult
    return KalimbaUnOp(op, regc, rega)


def kalimba_minim_decode_subword_a(instruction, op):
    rw_data_sel = get_bits(instruction, 9, 3)
    sel = KalimbaSubWordMem(rw_data_sel)
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    if rega == KalimbaBank1Reg.Null:
        rega = KalimbaBank3Reg.FP
    return KalimbaSubWordMemAccess(op, sel, regc, rega, regb)


def kalimba_minim_decode_subword_b(instruction, op):
    rw_data_sel = get_bits(instruction, 9, 3)
    sel = KalimbaSubWordMem(rw_data_sel)
    rega, regb, regc = kalimba_minim_decode_regs_a(instruction)
    if rega == KalimbaBank1Reg.Null:
        rega = KalimbaBank3Reg.FP
    K = (get_bits(instruction, 12, 2) << 3) + get_bits(instruction, 6, 3)
    return KalimbaSubWordMemAccess(op, sel, regc, rega, K)


def kalimba_minim_decode_subword_fp(instruction, op):
    rw_data_sel = get_bits(instruction, 9, 3)
    sel = KalimbaSubWordMem(rw_data_sel)
    regc = KalimbaBank1Reg(get_bits(instruction, 0, 3))
    rega = KalimbaBank3Reg.FP
    K = get_bits(instruction, 3, 6)
    return KalimbaSubWordMemAccess(op, sel, regc, rega, K)


minim_unprefixed_instructions = [
    (0b1111_111_000_000_000, 0b0000_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0000_010_000_000_000, KalimbaOp.SUB, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0000_001_000_000_000, KalimbaOp.ADC, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0000_011_000_000_000, KalimbaOp.SBB, kalimba_minim_decode_a_op_b_typea),  # A maybe
    (0b1111_111_111_000_000, 0b0000_100_000_000_000, KalimbaOp.SE8, kalimba_minim_decode_se8_16),
    (0b1111_111_111_000_000, 0b0000_110_000_000_000, KalimbaOp.SE16, kalimba_minim_decode_se8_16),
    (0b1111_100_000_000_000, 0b0000_100_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_mov_add_a),  # A
    (0b1110_010_000_000_000, 0b0010_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_add_sub_b),  # B
    (0b1110_010_000_000_000, 0b0010_010_000_000_000, KalimbaOp.SUB, kalimba_minim_decode_add_sub_b),  # B
    (0b1011_000_000_111_000, 0b0001_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_fp_adjust_b),  # B
    (0b1111_111_000_000_000, 0b0001_000_000_000_000, KalimbaOp.AND, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0001_001_000_000_000, KalimbaOp.OR, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0001_010_000_000_000, KalimbaOp.XOR, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0001_011_000_000_000, KalimbaOp.LSHIFT, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0001_100_000_000_000, KalimbaOp.ASHIFT, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_111_000_000_000, 0b0001_101_000_000_000, KalimbaOp.IMUL, kalimba_minim_decode_a_op_b_typea),  # A
    (0b1111_110_000_000_000, 0b0001_110_000_000_000, KalimbaOp.PUSHM, kalimba_minim_decode_pushm_popm),
    (0b1111_100_000_000_000, 0b0100_000_000_000_000, KalimbaOp.IMUL, kalimba_minim_decode_a_op_b_typeb),  # B
    (0b1111_110_000_000_000, 0b0100_100_000_000_000, KalimbaOp.POPM, kalimba_minim_decode_pushm_popm),
    (0b1111_111_111_000_000, 0b0100_110_000_000_000, KalimbaOp.DOLOOP, kalimba_minim_decode_call_jump_do_k),
    (0b1111_111_111_000_000, 0b0100_110_001_000_000, KalimbaOp.ADD, kalimba_minim_decode_sp_adjust),  # B sp adj k
    (0b1111_111_111_000_000, 0b0100_110_010_000_000, KalimbaOp.DIV, kalimba_minim_decode_div),
    (0b1111_111_111_110_000, 0b0100_110_011_000_000, KalimbaOp.DIV, kalimba_minim_decode_div_result),  # unop
    (0b1111_111_111_111_000, 0b0100_110_011_010_000, KalimbaOp.CALL, kalimba_minim_decode_call_jump_regc),  # regc
    (0b1111_111_111_111_000, 0b0100_110_011_011_000, KalimbaOp.JUMP, kalimba_minim_decode_call_jump_regc),  # regc
    (0b1111_111_000_000_000, 0b0100_111_000_000_000, KalimbaOp.CALL, kalimba_minim_decode_call_jump_do_k),  # k
    (0b1111_100_000_000_000, 0b0101_000_000_000_000, KalimbaOp.LSHIFT, kalimba_minim_decode_a_op_b_typeb),  # B
    (0b1111_100_000_000_000, 0b0101_100_000_000_000, KalimbaOp.ASHIFT, kalimba_minim_decode_a_op_b_typeb),  # B
    (0b1111_000_000_000_000, 0b0110_000_000_000_000, KalimbaOp.JUMP, kalimba_minim_decode_call_jump_do_k),  # k cond
    (0b1111_000_000_000_000, 0b0111_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_mov_add_b),  # B
    (0b1100_000_000_000_000, 0b1000_000_000_000_000, KalimbaOp.LOADW, kalimba_minim_decode_subword_b),  # B
    (0b1111_100_000_000_000, 0b1100_000_000_000_000, KalimbaOp.AND, kalimba_minim_decode_a_op_b_typeb),  # B
    (0b1111_100_000_000_000, 0b1100_100_000_000_000, KalimbaOp.OR, kalimba_minim_decode_a_op_b_typeb),  # B
    (0b1111_000_000_000_000, 0b1101_000_000_000_000, KalimbaOp.LOADW, kalimba_minim_decode_subword_fp),
    (0b1111_000_000_000_000, 0b1110_000_000_000_000, KalimbaOp.LOADW, kalimba_minim_decode_subword_a),  # A
    (0b1111_000_000_000_000, 0b1111_000_000_000_000, KalimbaOp.PREFIX, 0),
]


def kalimba_minim_decode_prefixed_and_b(instruction, op, prefixes):
    rega = KalimbaBank1Reg(get_bits(prefixes[-1], 4, 4))
    regc = KalimbaBank1Reg(get_bits(prefixes[-1], 0, 4))

    K13 = get_bits(instruction, 0, 13)
    K4 = get_bits(prefixes[-1], 8, 4)
    K = (K4 << 13) + K13

    main_code_and_prefix_len = 17
    pre_prefix_len = 12
    pre_pre_prefix_len = 5

    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    prefixes_len = len(prefixes)
    if prefixes_len >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if prefixes_len >= 3:
        sign_extended = False
        K += (prefixes[-3] << (main_code_and_prefix_len + pre_prefix_len))
        total_len += pre_pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
            sign_extended = True

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)

    return KalimbaBinOp(op, regc, rega, K)


def kalimba_minim_decode_prefixed_add_sub_b(instruction, op, prefixes):
    rega = reg_bank_lut[get_bits(
        prefixes[-1], 10, 1) + 1](get_bits(prefixes[-1], 4, 4))
    regc = reg_bank_lut[get_bits(
        prefixes[-1], 11, 1) + 1](get_bits(prefixes[-1], 0, 4))

    K = (get_bits(prefixes[-1], 8, 2) << 12) + \
        (get_bits(instruction, 11, 2) << 10) + get_bits(instruction, 0, 10)

    main_code_and_prefix_len = 14
    pre_prefix_len = 12
    pre_pre_prefix_len = 5
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    prefixes_len = len(prefixes)
    if prefixes_len >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if prefixes_len >= 3:
        sign_extended = False
        K += (prefixes[-3] << (main_code_and_prefix_len + pre_prefix_len))
        total_len += pre_pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
            sign_extended = True

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)

    return KalimbaBinOp(op, regc, rega, K)


def kalimba_minim_decode_prefixed_mov_add_b(instruction, op, prefixes):

    K = (get_bits(prefixes[-1], 4, 7) << 13) + \
        (get_bits(instruction, 11, 3) << 10) + get_bits(instruction, 0, 10)

    main_code_and_prefix_len = 20
    pre_prefix_len = 12
    pre_pre_prefix_len = 5
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    # diff :Table 6-27 Table 6-28
    # 80f7b1f87c6b -> I1 = 0x1777c
    # 80f7b1f87c6b -> I1 = 0x7801777C
    prefixes_len = len(prefixes)
    if prefixes_len >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)

    regc = reg_bank_lut[get_bits(
        prefixes[-1], 11, 1) + 1](get_bits(prefixes[-1], 0, 4))

    if get_bits(instruction, 10, 1) == 0:
        rega = KalimbaBank1Reg.Null
    else:
        rega = regc
    return KalimbaBinOp(op, regc, rega, K)


def kalimba_minim_decode_prefixed_subword_b(instruction, op, prefixes):

    K = (get_bits(prefixes[-1], 8, 2) << 11) + \
        (get_bits(instruction, 12, 2) << 9) + get_bits(instruction, 0, 9)

    main_code_and_prefix_len = 13
    pre_prefix_len = 12
    pre_pre_prefix_len = 7
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    prefixes_len = len(prefixes)
    if prefixes_len >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if prefixes_len >= 3:
        sign_extended = False
        K += (prefixes[-3] << (main_code_and_prefix_len + pre_prefix_len))
        total_len += pre_pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
            sign_extended = True
    rw_data_sel = get_bits(instruction, 9, 3)
    sel = KalimbaSubWordMem(rw_data_sel)

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)

    rega = reg_bank_lut[get_bits(
        prefixes[-1], 10, 1) + 1](get_bits(prefixes[-1], 4, 4))
    regc = reg_bank_lut[get_bits(
        prefixes[-1], 11, 1) + 1](get_bits(prefixes[-1], 0, 4))

    if rega == KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    return KalimbaSubWordMemAccess(op, sel, regc, rega, K)


def kalimba_minim_decode_prefixed_subword_a(instruction, op, prefixes):
    rega = reg_bank_lut[get_bits(
        prefixes[-1], 10, 1) + 1](get_bits(prefixes[-1], 4, 4))
    regc = reg_bank_lut[get_bits(
        prefixes[-1], 11, 1) + 1](get_bits(prefixes[-1], 0, 4))
    regb = reg_bank_lut[get_bits(
        instruction, 4, 1) + 1](get_bits(instruction, 0, 4))

    rw_data_sel = get_bits(instruction, 9, 3)
    sel = KalimbaSubWordMem(rw_data_sel)
    is_sub = False
    if get_bits(instruction, 8, 1) == 1:
        is_sub = True
    return KalimbaSubWordMemAccess(op, sel, regc, rega, regb, sub=is_sub)


def kalimba_minim_decode_prefixed_jump(instruction, op, prefixes):
    K = (get_bits(prefixes[-1], 4, 8) << 11) + \
        (get_bits(instruction, 8, 4) << 7) + get_bits(instruction, 0, 7)
    main_code_and_prefix_len = 19
    pre_prefix_len = 12
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    if len(prefixes) >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True
    if (K & 1) != 0:
        K -= 1
    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)
    cond = KalimbaCond(get_bits(instruction, 0, 4))
    return KalimbaControlFlow(op, K * 2, cond, None)


def kalimba_minim_decode_prefixed_call(instruction, op, prefixes):

    K = (prefixes[-1] << 9) + (get_bits(instruction, 8, 4)
                               << 5) + get_bits(instruction, 0, 5)
    main_code_and_prefix_len = 21
    pre_prefix_len = 8
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    cond = KalimbaCond.Always
    if len(prefixes) >= 2:
        sign_extended = False
        cond = KalimbaCond(get_bits(prefixes[-2], 0, 4))
        K += (get_bits(prefixes[-2], 4, pre_prefix_len)
              << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True
    if (K & 1) != 0:
        K -= 1
    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)
    return KalimbaControlFlow(op, K * 2, cond, None)


def kalimba_minim_decode_prefixed_push(instruction, op, prefixes):

    K = (prefixes[-1] << 8) + (get_bits(instruction, 8, 4)
                               << 4) + get_bits(instruction, 0, 4)
    main_code_and_prefix_len = 20
    pre_prefix_len = 8
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    if len(prefixes) >= 2:
        sign_extended = False
        K += (get_bits(prefixes[-2], 4, pre_prefix_len)
              << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)

    return KalimbaPushOff(op, KalimbaBank1Reg.Null, K)


# TODO POP RTS
def kalimba_minim_decode_prefixed_pushm_popm(instruction, op, prefixes):
    bank_sel = reg_bank_lut[get_bits(instruction, 10, 2) + 1]
    sp_adjust = get_bits(instruction, 8, 2) * 16
    bitfield = (prefixes[-1] << 4) + get_bits(instruction, 0, 4)
    reg_list = []
    for i in range(16):
        if ((bitfield >> i) & 1) == 1:
            reg_list.append(bank_sel(i))

    return KalimbaStackOp(op, reg_list, KalimbaCond.Always, sp_adjust)


def kalimba_minim_decode_prefixed_doloop(instruction, op, prefixes):

    K = (prefixes[-1] << 3) + get_bits(instruction, 0, 3)
    main_code_and_prefix_len = 15
    pre_prefix_len = 12
    total_len = main_code_and_prefix_len
    sign_extended = False
    if (K & (1 << (main_code_and_prefix_len - 1))) != 0:
        sign_extended = True
    if len(prefixes) >= 2:
        sign_extended = False
        K += (prefixes[-2] << main_code_and_prefix_len)
        total_len += pre_prefix_len
        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
            sign_extended = True

    if sign_extended:
        K = nbits_unsigned_to_signed(K, total_len)
    return KalimbaControlFlow(op, K * 2, KalimbaCond.Always, None)


def kalimba_minim_decode_prefixed_insert32(instruction, op, prefixes):
    maxim_instruction_type = get_bits(instruction, 4, 2)
    prefixes_len = len(prefixes)
    if maxim_instruction_type == 0b00:  # Type A
        maxim_instruction = (get_bits(instruction, 0, 12) << 20) + \
            (get_bits(prefixes[-1], 8, 4) << 16) + get_bits(prefixes[-1], 0, 8)
    elif maxim_instruction_type == 0b01:  # Type B
        if prefixes_len == 1:
            K9 = (get_bits(prefixes[-1], 0, 8) << 1) + \
                get_bits(instruction, 12, 1)
            K = K9
            if (K9 >> 8) == 1:
                K9 += 0xFE00  # 0xFE00
                K = K9 + 0xFFFF0000
            maxim_instruction = (get_bits(instruction, 0, 12)
                                 << 20) + (get_bits(prefixes[-1], 8, 4) << 16) + K9

        elif prefixes_len == 2 and get_bits(instruction, 12, 4) == 0b1101:
            K20 = (prefixes[-2] << 8) + get_bits(prefixes[-1], 0, 8)
            K = K20
            if (K20 >> 19) == 1:
                K = K20 + 0xFFF00000
            maxim_instruction = (get_bits(instruction, 0, 12) << 20) + \
                (get_bits(prefixes[-1], 8, 4) << 16) + (K & 0xffff)
            # TODO MaxiM Prefix

        elif prefixes_len == 3:
            K = (prefixes[-3] << 20) + (prefixes[-2] << 8) + \
                get_bits(prefixes[-1], 0, 8)
            maxim_instruction = (get_bits(instruction, 0, 12) << 20) + \
                (get_bits(prefixes[-1], 8, 4) << 16) + (K & 0xffff)
            # TODO MaxiM Prefix

    elif maxim_instruction_type == 0b10 or maxim_instruction_type == 0b11:
        if get_bits(instruction, 12, 4) == 0b1100:  # Type C INS
            maxim_instruction = (get_bits(instruction, 0, 12) << 20) + (
                get_bits(prefixes[-1], 8, 4) << 16) + (get_bits(prefixes[-1], 0, 8) << 8)

        if get_bits(instruction, 12, 4) == 0b1101:  # Type C AG
            maxim_instruction = (get_bits(instruction, 4, 8) << 24) + \
                (get_bits(instruction, 0, 4) << 12) + prefixes[-1]

    if prefixes_len == 2 and get_bits(instruction, 12, 4) == 0b1100:  # Normal ?
        maxim_instruction = (get_bits(instruction, 0, 12) << 20) + \
                            (get_bits(prefixes[-1], 8, 4) << 16) + \
                            (get_bits(prefixes[-2], 0, 8) << 8) + \
            get_bits(prefixes[-1], 0, 8)
    instruction, opcode, decode = kalimba_maxim_lookup_decoder(
        maxim_instruction)
    return decode(instruction, opcode, None)


minim_prefixed_instructions = [
    (0b1110_000_000_000_000, 0b0000_000_000_000_000, KalimbaOp.AND, kalimba_minim_decode_prefixed_and_b),  # B
    (0b1110_010_000_000_000, 0b0010_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_prefixed_add_sub_b),  # B
    (0b1110_010_000_000_000, 0b0010_010_000_000_000, KalimbaOp.SUB, kalimba_minim_decode_prefixed_add_sub_b),  # B
    (0b1100_000_000_000_000, 0b0100_000_000_000_000, KalimbaOp.ADD, kalimba_minim_decode_prefixed_mov_add_b),  # B
    (0b1100_000_000_000_000, 0b1000_000_000_000_000, KalimbaOp.LOADW, kalimba_minim_decode_prefixed_subword_b),  # B
    (0b1110_000_000_000_000, 0b1100_000_000_000_000, KalimbaOp.INSERT32, kalimba_minim_decode_prefixed_insert32),
    (0b1111_000_011_100_000, 0b1110_000_000_000_000, KalimbaOp.LOADW, kalimba_minim_decode_prefixed_subword_a),  # A
    (0b1111_000_010_000_000, 0b1110_000_010_000_000, KalimbaOp.JUMP, kalimba_minim_decode_prefixed_jump),  # B
    (0b1111_000_011_100_000, 0b1110_000_000_100_000, KalimbaOp.CALL, kalimba_minim_decode_prefixed_call),  # B
    (0b1111_000_011_110_000, 0b1110_000_001_010_000, KalimbaOp.PUSH, kalimba_minim_decode_prefixed_push),
    (0b1111_000_011_110_000, 0b1110_000_001_000_000, KalimbaOp.PUSHM, kalimba_minim_decode_prefixed_pushm_popm),
    (0b1111_000_011_100_000, 0b1110_000_001_100_000, KalimbaOp.POPM, kalimba_minim_decode_prefixed_pushm_popm),  # rts
    (0b1111_111_111_111_000, 0b1110_110_001_000_000, KalimbaOp.DOLOOP, kalimba_minim_decode_prefixed_doloop),
    (0b1111_000_000_000_000, 0b1111_000_000_000_000, KalimbaOp.PREFIX, 0),
]


def kalimba_maxim_decode(data: bytes, addr: int):
    prefixes = []
    offset = 0
    if len(data) < 2:
        return 0
    (instruction, ) = unpack('<H', data[offset:offset+2])
    # log_info(f'instr :{hex(instr)} len{len(data)}')
    offset += 2
    opcode = get_bits(instruction, 12, 4)

    while opcode == 0b1111:  # PREFIX
        prefixes.append(get_bits(instruction, 0, 12))
        if len(data) < offset + 2:
            return 0
        (instruction, ) = unpack('<H', data[offset:offset+2])
        offset += 2
        opcode = get_bits(instruction, 12, 4)

    len_prefixes = len(prefixes)
    print(hex(addr), end=' ')

    if len_prefixes == 0:
        # print(f'{instr:b}')
        for mask, value, op, func in minim_unprefixed_instructions:
            if (instruction & mask) == value:
                # print(op.name)
                print(func(instruction, op).__str__())

                break
    else:
        for mask, value, op, func in minim_prefixed_instructions:
            if (instruction & mask) == value:
                # print(op.name)
                print(func(instruction, op, prefixes).__str__())
                break
    return offset


if __name__ == '__main__':
    import os
    os.system('clear')
    with open('flash_image.xuv_apps_p1.bin', 'rb') as f:
        f.seek(0x180)
        data = f.read(0xcbf4 - 0x180)
        length = 0
        while True:
            inc = kalimba_maxim_decode(data[length:], length + 0x180)
            if inc == 0:
                break
            length += inc
