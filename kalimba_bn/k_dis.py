from struct import unpack
from .k_instr import *
bank1_2_3 = [
    'Null','rMAC','r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','rLink','rFlags','rMACB',
    'I0','I1','I2','I3','I4','I5','I6','I7','M0','M1','M2','M3','L0','L1','L4','L5',
    'rMAC2','rMAC1','rMAC0','DoLoopStart','DoLoopEnd','DivResult','DivRemainder','rMACB2','rMACB1','rMACB0','B0','B1','B4','B5','FP','SP'
]
cond_3bit_map_to_4bit = [0,1,3,10,11,12,13,15]

cond_4bit = [
    'Z/EQ', 'NZ/NE',
    'C/NB', 'NC/B',
    'NEG', 'POS',
    'V', 'NV',
    'HI', 'LS',
    'GE', 'LT',
    'GT', 'LE',
    'USERDEF', 'Always'
]

pushm_reg_bitmap = [
    0x0000,0x0002,0x0006,0x000e,0x001e,0x003e,0x007e,0x00fe,
    0x0100,0x0004,0x000c,0x001c,0x003c,0x007c,0x00fc,0x01fc,
    0x0200,0x0008,0x0018,0x0038,0x0078,0x00f8,0x01f8,0x03f8,
    0x0400,0x0010,0x0030,0x0070,0x00f0,0x01f0,0x03f0,0x07f0,
    0x0800,0x0020,0x0060,0x00e0,0x01e0,0x03e0,0x07e0,0x0fe0,
    0x1000,0x0040,0x00c0,0x01c0,0x03c0,0x07c0,0x0fc0,0x1fc0,
    0x8000,0x0080,0x0180,0x0380,0x0780,0x0f80,0x1f80,0x3f80]
#no XOR?
k5_logical_immediate = [
    0x1,0x2,0x3,0x4,0x7,0x8,0xF,0x10,
    0x1F,0x20,0x3F,0x40,0x7F,0x80,0xFF,0x100,
    0x1FF,0x200,0x3FF,0x400,0x7FF,0x800,0xFFF,0x1000,
    0x7FFF,0x8000,0xFFFF,0x10000,0xFFFFF,0x7FF0000,0x7FFFFFF,0x80000000
]
k5_shift_immediate = [
    -1,-2,-3,-4,-5,-6,-7,-8,
    -12,-14,-16,-20,-22,-24,-31,-32,
    1,2,3,4,5,6,7,8,
    12,14,16,20,22,24,31,32,
]
def get_mask(length):
    return ((1<<(length))-1)
def get_bits(instruction,offset,length):
    return (instruction>>offset)&get_mask(length)
def get_5bit_reg(index):
    return bank1_2_3[index]
def get_4bit_reg(index):
    return bank1_2_3[index]
def get_3bit_reg(index):
    return bank1_2_3[index]
def get_3bit_cond(index):
    return cond_4bit[cond_3bit_map_to_4bit[index]]
def get_4bit_cond(index):
    return cond_4bit[index]
#TODO rw_data_sel size Table 5-2 Table 5-3
def get_rw_data_sel(rw_data_sel):
    return rw_data_instructions[rw_data_sel]
def nbits_unsigned_to_signed(x, bits):
    if (x & (1 << (bits- 1))) != 0:
        return -(((~x)&((1<<bits)-1))+1)
    else:
        return x

def get_type_b_disassembly_description(opcode, regc, rega, k, description:kalimba_minim_instr):
    #print(opcode)
    description.instr_type = kalimba_minim_instr_type.TYPE_B
    description.rega = get_4bit_reg(rega)
    description.regc = get_4bit_reg(regc)
    description.regb_k = k
    if (opcode >> 4) == 0b00:
        if ((opcode>>3)&1) == 1:
            pass
    elif opcode == 0b100000:
        description.op = 'AND'
    elif opcode == 0b100001:
        description.op = 'OR'
    elif opcode == 0b100010:
        description.op = 'XOR'
    elif opcode == 0b100011:
        description.op = 'LSHIFT'
    elif opcode == 0b100100:
        description.op = 'ASHIFT'
    elif opcode == 0b110100:
        description.op = '=M'
        description.regb_k = k #unsigned?
    elif opcode == 0b111010:
        description.op = 'LSHIFT'
        param = al_instructions_param()
        param.shift_reverse = True
        description.param = param
    elif opcode == 0b111011:
        description.op = 'ASHIFT'
        param = al_instructions_param()
        param.shift_reverse = True
        description.param = param
    elif opcode == 0b111100:#Table 6-17
        options = rega >> 2
        bank_sel = rega & 0b11
        if bank_sel != 0b11:
            pass
        else:
            description.regb_k = k
            if k >= 0:
                description.op = '+'
            else:
                description.op = '-'
            
            if options == 0b00:
                if regc == 0:
                    description.regc = 'SP'
                    description.rega = 'SP'
                else:
                    description.regc = get_4bit_reg(regc)
                    description.rega = 'SP'
            elif options == 0b01:
                if regc == 0:
                    description.regc = 'FP'
                    description.rega = 'FP'
                else:
                    description.regc = get_4bit_reg(regc)
                    description.rega = 'FP'
            elif options == 0b11:
                description.op = 'pushc'
                description.regc = get_4bit_reg(regc)
    return
def get_disassembly_description(data: bytes, addr: int):
    prefixs = []
    offset = 0
    if len(data) < 2:
        return None
    (instr, ) = unpack('<H',data[offset:offset+2])
    #log_info(f'instr :{hex(instr)} len{len(data)}')
    offset += 2
    opcode = get_bits(instr,12,4)
    
    while opcode == 0b1111:# PREFIX
        prefixs.append(get_bits(instr,0,12))
        if len(data) < offset + 2:
            return None
        (instr, ) = unpack('<H',data[offset:offset+2])
        offset += 2
        opcode = get_bits(instr, 12, 4)

    len_prefixs = len(prefixs)
    description = kalimba_minim_instr()
    description.length = offset
    if len_prefixs == 0:#Unprefixed
        options = get_bits(instr,9,3)
        regb = get_bits(instr,6,3)
        rega = get_bits(instr,3,3)
        regc = get_bits(instr,0,3)
        if opcode == 0b0000:# ADD/SUB (A) SE8 SE16 MOV/ADD (A)
            if (options & 0b100) == 0b000:#ADD/SUB (A)
                description.regc = get_3bit_reg(regc)
                description.rega = get_3bit_reg(rega)
                if (options & 0b010) == 0b000:
                    description.op = '+'
                else:
                    description.op = '-'
                description.regb_k = get_3bit_reg(regb)
                param = al_instructions_param()
                param.use_carry = True
                description.param = param

            elif options == 0b100 and regb == 0:#SE8
                description.instr_type = kalimba_minim_instr_type.TYPE_NO_REGB_K
                description.regc = get_3bit_reg(regc)
                description.rega = get_3bit_reg(rega)
                description.op = 'SE8'


            elif options == 0b110 and regb == 0:#SE16
                description.instr_type = kalimba_minim_instr_type.TYPE_NO_REGB_K
                description.regc = get_3bit_reg(regc)
                description.rega = get_3bit_reg(rega)
                description.op = 'SE16'


            else:#MOV/ADD (A)
                RegCBank = (get_bits(instr,8,2)<<3)+regc
                RegABank = (get_bits(instr,6,2)<<3)+rega
                description.regc = get_5bit_reg(RegCBank)
                if get_bits(instr,10,1) == 1:
                    description.rega = description.regc
                    description.op = '+'
                else:
                    description.op = 'mv'
                description.regb_k = get_5bit_reg(RegABank)


        elif (opcode & 0b1110) == 0b0010:# ADD/SUB (B)
            K = (get_bits(instr, 11, 2) << 4) + get_bits(instr, 6, 4)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regc = get_3bit_reg(regc)
            description.rega = get_3bit_reg(rega)
            if (options & 0b010) == 0b000:
                description.op = '+'
            else:
                description.op = '-'
            description.regb_k = K


        elif (opcode & 0b1011) == 0b0001 and rega == 0:# Reg=FP+K
            K = (get_bits(instr,14,1) << 6) + get_bits(instr,6,6)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regc = get_3bit_reg(regc)
            description.rega = 'FP'
            description.op = '+'
            description.regb_k = K * 4


        elif opcode == 0b0001:# AND (A),OR (A),XOR (A),LSHIFT (A),ASHIFT (A),IMULT (A), PUSHM
            description.regc = get_3bit_reg(regc)
            description.rega = get_3bit_reg(rega)
            description.regb_k = get_3bit_reg(regb)
            if options == 0b000:
                description.op = 'AND'
            elif options == 0b001:
                description.op = 'OR'
            elif options == 0b010:
                description.op = 'XOR'
            elif options == 0b011:
                description.op = 'LSHIFT'
            elif options == 0b100:
                description.op = 'ASHIFT'
            elif options == 0b101:
                description.op = '*'


            else:#PUSHM
                reg_bitmap = pushm_reg_bitmap[get_bits(instr,0,6) - 8]
                FP = get_bits(instr,6,1)
                rLink = get_bits(instr,7,1)
                param = stack_instructions_param()
                SP_adj = get_bits(instr,8,2)
                description.op = 'pushm'
                if FP == 1:
                    param.reg_list.append('FP')
                for i in range(16):
                    if ((reg_bitmap >> i) & 1) != 0:
                        param.reg_list.append(get_3bit_reg(i))
                if rLink == 1:
                    param.reg_list.append('rLink')
                param.sp_adjust = SP_adj * 16
                description.param = param


        elif opcode == 0b0100:
            if (options & 0b100) == 0b000:#IMULT (B)
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                description.regc = get_3bit_reg(regc)
                description.rega = get_3bit_reg(rega)
                description.regb_k = get_bits(instr,6,5)
                description.op = '*'


            elif (options & 0b110) == 0b100:# POPM
                reg_bitmap = pushm_reg_bitmap[get_bits(instr,0,6) - 8]
                FP = get_bits(instr,6,1)
                rLink = get_bits(instr,7,1)
                SP_adj = get_bits(instr,8,2)
                description.op = 'popm'
                param = stack_instructions_param()
                if FP == 1:
                    param.reg_list.append('FP')
                for i in range(16):
                    if ((reg_bitmap >> i) & 1) != 0:
                        param.reg_list.append(get_3bit_reg(i))
                if rLink == 1:
                    param.reg_list.append('rLink')
                param.sp_adjust = -SP_adj * 16
                description.param = param


            elif options == 0b110 and regb == 0b000:# DOLOOP
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                description.op = 'do'
                description.regb_k = get_bits(instr, 0, 6)#if addr: 5d4 k6u: 2, loop to 5d8
                if (description.regb_k & 1) != 0:
                    description.op = 'do(m)'
                    description.regb_k -= 1
                description.regb_k *= 2
                description.regb_k += 2
            elif options == 0b110 and regb == 0b001:# SP=SP+K
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                description.regc = 'SP'
                description.rega = 'SP'
                description.regb_k = nbits_unsigned_to_signed(get_bits(instr,0,6),6)
                description.regb_k *= 4
                description.op = '+'


            elif options == 0b110 and regb == 0b010:#Div = RegC/RegA
                description.regc = 'Div'
                description.rega = get_3bit_reg(regc)
                description.regb_k = get_3bit_reg(rega)
                description.op = '/'


            elif options == 0b110 and regb == 0b011 and (rega & 0b110) == 0b000:#RegC=DivRes or Rem
                description.op = 'mv'
                description.regc = get_3bit_reg(regc)
                if get_bits(instr,3,1) == 1:
                    description.rega = 'DivRemainder'
                else:
                    description.rega = 'DivResult'


            elif options == 0b110 and regb == 0b011 and (rega & 0b111) == 0b010:#CALL RegC
                description.op = 'call'
                description.regc = get_3bit_reg(regc)


            elif options == 0b110 and regb == 0b011 and (rega & 0b111) == 0b011:#JUMP RegC
                if regc == 0:
                    description.op = 'rts'
                else:
                    description.op = 'jump'
                    description.regc = get_3bit_reg(regc)


            elif options == 0b111:#CALL K9
                description.op = 'call(m)' # FIXME: is this correct?
                k9u = get_bits(instr, 0, 9)
                description.regb_k = nbits_unsigned_to_signed(k9u, 9) * 2
                description.instr_type = kalimba_minim_instr_type.TYPE_B


        elif opcode == 0b0101:#LSHIFT (B) ASHIFT(B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regb_k = k5_shift_immediate[get_bits(instr,6,5)]
            description.regc = get_3bit_reg(regc)
            description.rega = get_3bit_reg(rega)
            if get_bits(instr,11,1) == 0:
                description.op = 'LSHIFT'
            else:
                description.op = 'ASHIFT'


        elif opcode == 0b0110:#JUMP K9 (cond)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.op = 'jump(m)'
            description.regb_k = nbits_unsigned_to_signed(get_bits(instr, 0, 9), 9) * 2
            param = program_flow_instructions_param()
            param.cond = get_3bit_cond(options)
            description.param = param


        elif opcode == 0b0111:#MOV/ADD (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regc = get_5bit_reg((get_bits(instr,8,2)<<3)+regc)
            if get_bits(instr,10,1) == 1:
                description.rega = description.regc
                description.op = '+'
            else:
                description.op = 'mv'
            sig = (get_bits(instr,11,1) << 5)
            description.regb_k = nbits_unsigned_to_signed(sig + get_bits(instr, 3, 5), 6)



        elif (opcode & 0b1100) == 0b1000:#Subword (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regb_k = (get_bits(instr,12,2)<<3) + regb
            if rega == 0:
                description.rega = 'FP'
            else:
                description.rega = get_3bit_reg(rega)
            description.regc = get_3bit_reg(regc)
            rw_data_sel = get_bits(instr,9,3)
            description.op = get_rw_data_sel(rw_data_sel)


        elif opcode == 0b1100:#AND (B) OR (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.regc = get_3bit_reg(regc)
            description.rega = get_3bit_reg(rega)
            description.regb_k = k5_logical_immediate[get_bits(instr,6,5)]
            if get_bits(instr,11,1) == 0:
                description.op = 'AND'
            else:
                description.op = 'OR'


        elif opcode == 0b1101:#FP mem access
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            description.rega = 'FP'
            description.regc = get_3bit_reg(regc)
            description.regb_k = get_bits(instr, 3, 6)
            rw_data_sel = get_bits(instr, 9, 3)
            description.op = get_rw_data_sel(rw_data_sel)
            if 'H' in description.op:
                description.regb_k *= 2
            elif 'B' in description.op:
                pass
            else:
                description.regb_k *= 4


        elif opcode == 0b1110:#Subword ADD (A)
            description.rega = get_3bit_reg(rega)
            '''
            if description.rega == 'Null':
                description.rega = 'FP'
            '''
            description.regc = get_3bit_reg(regc)
            description.regb_k = get_3bit_reg(regb)
            rw_data_sel = get_bits(instr, 9, 3)
            description.op = get_rw_data_sel(rw_data_sel)

    else:#prefixed
        prefix = prefixs[-1]
        if len_prefixs >= 2:
            pre_prefix = prefixs[-2]
        if len_prefixs >= 3:
            pre_pre_prefix = prefixs[-3]
        rega = get_bits(prefix,4,4)
        regc = get_bits(prefix,0,4)
        if (opcode & 0b1110) == 0b0000:#AND (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B
            #log_info('AND (B)')
            K13 = get_bits(instr, 0, 13)
            K4 = get_bits(prefix,8,4)
            K = (K4 << 13) + K13

            main_code_and_prefix_len = 17
            pre_prefix_len = 12
            pre_pre_prefix_len = 5
            total_len = main_code_and_prefix_len
            sign_extended = False
            if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                sign_extended = True

            if len_prefixs >= 2:
                sign_extended = False
                K += (pre_prefix << main_code_and_prefix_len)
                total_len += pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if len_prefixs >= 3:
                sign_extended = False
                K += (pre_pre_prefix << (main_code_and_prefix_len + pre_prefix_len))
                total_len += pre_pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if total_len < 32:
                total_len = 32

            if sign_extended:
                for i in range(total_len):
                    if (K & (1 << (total_len - i))) == 0:
                        K += (1 << (total_len - i))
                    else:
                        break
            description.regb_k = K
            #TODO >MAX32bit
            description.op = 'AND'
            description.regc = get_5bit_reg(regc)
            description.rega = get_5bit_reg(rega)


        elif (opcode & 0b1110) == 0b0010:#ADD/SUB (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B

            K = (get_bits(prefix, 8, 2) << 12) + (get_bits(instr, 11, 2) << 10 ) + get_bits(instr, 0, 10)

            main_code_and_prefix_len = 14
            pre_prefix_len = 12
            pre_pre_prefix_len = 5
            total_len = main_code_and_prefix_len
            sign_extended = False
            if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                sign_extended = True

            if len_prefixs >= 2:
                sign_extended = False
                K += (pre_prefix << main_code_and_prefix_len)
                total_len += pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if len_prefixs >= 3:
                sign_extended = False
                K += (pre_pre_prefix << (main_code_and_prefix_len + pre_prefix_len))
                total_len += pre_pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if total_len < 32:
                total_len = 32

            if sign_extended:
                for i in range(total_len):
                    if (K & (1 << (total_len - i))) == 0:
                        K += (1 << (total_len - i))
                    else:
                        break
            description.regb_k = K

            if get_bits(instr, 10, 1) == 1:
                description.op = '-'
            else:
                description.op = '+'
            regc += get_bits(prefix, 11, 1) << 4
            rega += get_bits(prefix, 10, 1) << 4
            description.regc = get_5bit_reg(regc)
            description.rega = get_5bit_reg(rega)


        elif (opcode & 0b1100) == 0b0100 and regc + (get_bits(prefix, 11, 1) << 4) != 0:#MOV/ADD (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B

            K = (get_bits(prefix, 4, 7) << 13) + (get_bits(instr, 11, 3) << 10 ) + get_bits(instr, 0, 10)

            main_code_and_prefix_len = 20
            pre_prefix_len = 12
            pre_pre_prefix_len = 5
            total_len = main_code_and_prefix_len
            sign_extended = False
            if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                sign_extended = True
            #diff :Table 6-27 Table 6-28
            # 80f7b1f87c6b -> I1 = 0x1777c
            # 80f7b1f87c6b -> I1 = 0x7801777C

            if len_prefixs >= 2:
                sign_extended = False
                K += (pre_prefix << main_code_and_prefix_len)
                total_len += pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if total_len < 32:
                total_len = 32

            if sign_extended:
                for i in range(total_len):
                    if (K & (1 << (total_len - i))) == 0:
                        K += (1 << (total_len - i))
                    else:
                        break
            description.regb_k = K
            regc += get_bits(prefix, 11, 1) << 4
            description.regc = get_5bit_reg(regc)
            if get_bits(instr, 10, 1) == 1:
                description.rega = description.regc
                description.op = '+'
            else:
                description.op = 'mv'
            

        elif (opcode & 0b1100) == 0b1000:#Subword (B)
            description.instr_type = kalimba_minim_instr_type.TYPE_B

            rw_data_sel = get_bits(instr,9,3)

            K = (get_bits(prefix,8,2)<<11) + (get_bits(instr,12,2)<<9) + get_bits(instr,0,9)

            main_code_and_prefix_len = 13
            pre_prefix_len = 12
            pre_pre_prefix_len = 7
            total_len = main_code_and_prefix_len
            sign_extended = False
            if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                sign_extended = True

            if len_prefixs >= 2:
                sign_extended = False
                K += (pre_prefix << main_code_and_prefix_len)
                total_len += pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                    sign_extended = True

            if len_prefixs >= 3:
                sign_extended = False
                K += (pre_pre_prefix << (main_code_and_prefix_len + pre_prefix_len))
                total_len += pre_pre_prefix_len
                if (K & (1 << (main_code_and_prefix_len + pre_prefix_len + pre_pre_prefix_len - 1))) != 0:
                    sign_extended = True

            description.op = get_rw_data_sel(rw_data_sel)
            if 'H' in description.op:
                total_len += 1
                K *= 2
            elif 'B' in description.op:
                pass
            else:
                total_len += 2
                K *= 4

            total_len += 2
            if total_len < 32:
                total_len = 32

            if sign_extended:
                for i in range(total_len):
                    if (K & (1 << (total_len - i))) == 0:
                        K += (1 << (total_len - i))
                    else:
                        break

            description.regb_k = K
            
            regc += get_bits(prefix, 11, 1) << 4
            rega += get_bits(prefix, 10, 1) << 4

            if rega == 14: # 'rFlags'
                rega = 14 + (2 << 4)

            description.regc = get_5bit_reg(regc)
            description.rega = get_5bit_reg(rega)
        
        
        elif (opcode & 0b1110) == 0b1100:#INSERT32
            description.is_insert32 = True
            description.op = 'INSERT32 '
            maxm_instr_type = get_bits(instr, 4, 2)
            opcode = get_bits(instr, 6, 6)
            if maxm_instr_type == 0b00:#type A
                maxm_instr = (get_bits(instr, 0, 12) << 20) + (get_bits(prefix, 8, 4) << 16) + get_bits(prefix, 0, 8)
                #print('a')
                description.op += f'A {opcode:b}'
            if maxm_instr_type == 0b01:#type B
                #print('b')
                description.op += f'B {opcode:b}'
                
                regc = get_bits(instr, 0, 4)
                rega = get_bits(prefix, 8, 4)
                if len_prefixs == 1:
                    K9 = (get_bits(prefix, 0, 8) << 1) + get_bits(instr, 12, 1)
                    K = K9
                    if (K9 >> 8) == 1:
                        K9 += 0xFE00#0xFE00
                        K = K9 + 0xFFFF0000
                    #maxm_instr = (get_bits(instr, 0, 12) << 20) + (get_bits(prefix, 8, 4) << 16) + K9
                    #print(hex(maxm_instr))
                elif len_prefixs == 2:
                    K20 = (pre_prefix << 8) + get_bits(prefix, 0, 8)
                    K = K20
                    if (K20 >> 19) == 1:
                        K = K20 + 0xFFF00000
                elif len_prefixs == 3:
                    K = (pre_pre_prefix << 20) + (pre_prefix << 8) + get_bits(prefix, 0, 8)
                get_type_b_disassembly_description(opcode, regc, rega, K, description)
            if maxm_instr_type == 0b10:#type C
                #print('c')
                description.op += f'C {opcode:b}'
                pass


            
        elif (opcode & 0b1111) == 0b1110:
            options2 = get_bits(instr,4,4)
            if (options2&0b1110) == 0b0000:#Subword mem ADD
                param = rw_data_instructions_param()
                description.regb_k = get_5bit_reg(get_bits(instr,0,5))
                if get_bits(prefix, 8, 1) == 0:
                    param.add = False

                rw_data_sel = get_bits(instr,9,3)
                description.op = get_rw_data_sel(rw_data_sel)
                regc += get_bits(prefix, 11, 1) << 4
                rega += get_bits(prefix, 10, 1) << 4    
                if rega == 14: # 'rFlags'
                    rega = 14 + (2 << 4)
                description.regc = get_5bit_reg(regc)
                description.rega = get_5bit_reg(rega)
                description.param = param


            elif (options2&0b1000) == 0b1000:#JUMP (B)
                description.op = 'jump'
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                K = (get_bits(prefix, 4, 8) << 11) + (get_bits(instr, 8, 4) << 7) + get_bits(instr, 0, 7)
                main_code_and_prefix_len = 19
                pre_prefix_len = 12
                total_len = main_code_and_prefix_len
                sign_extended = False
                if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                    sign_extended = True
                if len_prefixs >= 2:
                    sign_extended = False
                    K += (pre_prefix << main_code_and_prefix_len)
                    total_len += pre_prefix_len
                    if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                        sign_extended = True
                if (K & 1) != 0:
                    description.op = 'jump(m)'
                    K -= 1
                '''
                if total_len < 32:
                    total_len = 32

                if sign_extended:
                    for i in range(total_len):
                        if (K & (1 << (total_len - i))) == 0:
                            K += (1 << (total_len - i))
                        else:
                            break
                '''
                param = program_flow_instructions_param()
                param.cond = get_4bit_cond(regc)
                description.param = param
                description.regb_k = nbits_unsigned_to_signed(K, total_len)


            elif (options2&0b1110) == 0b0010:#CALL (B)
                description.op = 'call'
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                param = program_flow_instructions_param()
                K = (prefix << 9) + (get_bits(instr, 8, 4) << 5) + get_bits(instr, 0, 5)
                main_code_and_prefix_len = 21
                pre_prefix_len = 8
                total_len = main_code_and_prefix_len
                sign_extended = False
                if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                    sign_extended = True
                if len_prefixs >= 2:
                    sign_extended = False
                    param.cond = get_4bit_cond(get_bits(pre_prefix, 0, 4))
                    K += (get_bits(pre_prefix, 4, pre_prefix_len) << main_code_and_prefix_len)
                    total_len += pre_prefix_len
                    if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                        sign_extended = True
                if (K & 1) != 0:
                    description.op = 'call(m)'
                    K -= 1
                '''
                if total_len < 32:
                    total_len = 32
                
                if sign_extended:
                    for i in range(total_len):
                        if (K & (1 << (total_len - i))) == 0:
                            K += (1 << (total_len - i))
                        else:
                            break
                '''

                description.param = param
                description.regb_k = nbits_unsigned_to_signed(K, total_len)


            elif (options2&0b1111) == 0b0101:#PUSH Constant
                description.instr_type = kalimba_minim_instr_type.TYPE_B
                description.op = 'pushc'
                K = (prefix << 8) + (get_bits(instr, 8, 4) << 4) + get_bits(instr, 0, 4)
                main_code_and_prefix_len = 20
                pre_prefix_len = 8
                total_len = main_code_and_prefix_len
                sign_extended = False
                if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                    sign_extended = True
                if len_prefixs >= 2:
                    sign_extended = False
                    K += (get_bits(pre_prefix, 4, pre_prefix_len) << main_code_and_prefix_len)
                    total_len += pre_prefix_len
                    if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                        sign_extended = True

                if total_len < 32:
                    total_len = 32

                if sign_extended:
                    for i in range(total_len):
                        if (K & (1 << (total_len - i))) == 0:
                            K += (1 << (total_len - i))
                        else:
                            break
                description.regb_k = K


            elif (options2&0b1111) == 0b0100:
                bank_sel = get_bits(instr,10,2)
                if bank_sel == 0b11:#DOLOOP
                    description.op = 'do'
                    description.instr_type = kalimba_minim_instr_type.TYPE_B
                    K = (prefix << 3) + get_bits(instr, 0, 3)
                    main_code_and_prefix_len = 15
                    pre_prefix_len = 12
                    total_len = main_code_and_prefix_len
                    sign_extended = False
                    if (K & (1<<(main_code_and_prefix_len - 1))) != 0:
                        sign_extended = True
                    if len_prefixs >= 2:
                        sign_extended = False
                        K += (pre_prefix << main_code_and_prefix_len)
                        total_len += pre_prefix_len
                        if (K & (1 << (main_code_and_prefix_len + pre_prefix_len - 1))) != 0:
                            sign_extended = True

                    if total_len < 32:
                        total_len = 32

                    if sign_extended:
                        for i in range(total_len):
                            if (K & (1 << (total_len - i))) == 0:
                                K += (1 << (total_len - i))
                            else:
                                break
                    description.regb_k = K


                else:#PUSHM
                    description.op = 'pushm'
                    param = stack_instructions_param()
                    param.sp_adjust = get_bits(instr, 8, 2) * 16
                    bitfield = (prefix << 4) + get_bits(instr,0,4)
                    for i in range(16):
                        if ((bitfield >> i) & 1) == 1:
                            param.reg_list.append(bank1_2_3[bank_sel * 16 + i])
                    description.param = param

                    
            elif (options2&0b1110) == 0b0110:#POPM and RTS
                description.op = 'popm_rts'
                bank_sel = get_bits(instr,10,2)
                param = stack_instructions_param()
                param.sp_adjust = get_bits(instr, 8, 2) * 16
                bitfield = (prefix << 4) + get_bits(instr,0,4)
                for i in range(16):
                    if ((bitfield >> i) & 1) == 1:
                        param.reg_list.append(bank1_2_3[bank_sel * 16 + i])
                description.param = param


    return description
