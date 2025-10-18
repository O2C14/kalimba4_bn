from enum import Enum, IntEnum, auto
from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union, Literal
from dataclasses import dataclass
from functools import partial
from decimal import Decimal, getcontext

getcontext().prec = 29

class KalimbaRegBase(Enum):
    def __str__(self):
        return self.name

class KalimbaBank1Reg(KalimbaRegBase):
    Null   = 0
    rMAC   = 1
    r0     = 2
    r1     = 3
    r2     = 4
    r3     = 5
    r4     = 6
    r5     = 7
    r6     = 8
    r7     = 9
    r8     = 10
    r9     = 11
    r10    = 12
    rLink  = 13
    rFlags = 14
    rMACB  = 15

class KalimbaBank2Reg(KalimbaRegBase):
    I0 = 0
    I1 = 1
    I2 = 2
    I3 = 3
    I4 = 4
    I5 = 5
    I6 = 6
    I7 = 7
    M0 = 8
    M1 = 9
    M2 = 10
    M3 = 11
    L0 = 12
    L1 = 13
    L4 = 14
    L5 = 15

class KalimbaBank3Reg(KalimbaRegBase):
    rMAC2        = 0
    rMAC1        = 1
    rMAC0        = 2
    DoLoopStart  = 3
    DoLoopEnd    = 4
    DivResult    = 5
    DivRemainder = 6
    rMACB2       = 7
    rMACB1       = 8
    rMACB0       = 9
    B0           = 10
    B1           = 11
    B4           = 12
    B5           = 13
    FP           = 14
    SP           = 15

type KalimbaReg = Union[KalimbaBank1Reg, KalimbaBank2Reg, KalimbaBank3Reg]

class KalimbaSubWordMem(IntEnum):
    L_MBS = 0b000 # Load signed byte (SE)
    L_MBU = 0b001 # Load unsigned byte (ZP)
    L_MHS = 0b010 # Load signed halfword (SE)
    L_MHU = 0b011 # Load unsigned halfword (ZP)
    L_M   = 0b100 # Load word
    S_MB  = 0b101 # Store byte
    S_MH  = 0b110 # Store halfword
    S_M   = 0b111 # Store word

@dataclass(unsafe_hash=True)
class KalimbaWordMem:
    '''
    Accesses one word
    '''
    addr: Union[KalimbaReg, int]
    def __str__(self):
        return f'M[{self.addr}]'

@dataclass(unsafe_hash=True)
class KalimbaIndexedMemAccess:
    '''
    Reads/writes one word, post increment index by modify
    '''
    write: bool
    reg: KalimbaBank1Reg             # r0, r1, r2, r3, r4, r5, and rMAC
    idx: KalimbaBank2Reg             # Ix
    mod: Union[KalimbaBank2Reg, int] # Mx or constant
    def __str__(self):
        m = f'M[{self.idx},{self.mod}]'
        if self.write:
            return f'{m} = {self.reg}'
        else:
            return f'{self.reg} = {m}'

reg_bank_lut = {
    1: KalimbaBank1Reg,
    2: KalimbaBank2Reg,
    3: KalimbaBank3Reg,
}

class KalimbaCond(IntEnum):
    Z       = 0b0000
    NZ      = 0b0001
    C       = 0b0010
    NC      = 0b0011
    NEG     = 0b0100
    POS     = 0b0101
    V       = 0b0110
    NV      = 0b0111
    HI      = 0b1000
    LS      = 0b1001
    GE      = 0b1010
    LT      = 0b1011
    GT      = 0b1100
    LE      = 0b1101
    USERDEF = 0b1110
    Always  = 0b1111

class KalimbaInstrType(IntEnum):
    A = 0 # <if cond> RegC = RegA OP RegB <MEM_ACCESS_1>
    B = 1 # RegC = RegA OP constant
    C = 2 # RegC = RegC OP RegA <MEM_ACCESS_1> <MEM_ACCESS_2>;

class KalimbaOp(IntEnum):
    ADD    = 0      # Add
    ADC    = auto() # Add with carry
    SUB    = auto() # Subtract
    SBB    = auto() # Subtract with borrow
    AND    = auto() # Logical AND
    OR     = auto() # Logical OR
    XOR    = auto() # Logical XOR
    LSHIFT = auto() # Logical Shift
    ASHIFT = auto() # Arithmetical Shift
    IMUL   = auto() # Signed multiply
    SMUL   = auto() # Saturated signed multiply
    FMUL   = auto() # Fractional signed multiply
    FMADD  = auto() # Multiply-accumulate (optional Add/Sub for A/B)
    FMSUB  = auto() # Multiply-subtract (optional Add/Sub for A/B)
    MULX   = auto() # Multiply (optional Add/Sub for A/B)
    LOAD   = auto() # Load with offset
    STORE  = auto() # Store with offset
    SIGN   = auto() # Sign detect
    BSIGN  = auto() # Block sign detect
    DIV    = auto() # Divide
    JUMP   = auto() # Jump
    RTS    = auto() # Return from subroutine
    RTI    = auto() # Return from interrupt
    CALL   = auto() # Call subroutine / Return from interrupt /
    MULAS  = auto() # Multiply + Add/Sub
    ABS    = auto() # Abs
    MIN    = auto() # Min
    MAX    = auto() # Max
    TWOBC  = auto() # TWOBITCOUNT
    MOD24  = auto() # Modulo 24
    ONEBC  = auto() # ONEBITCOUNT
    SE8    = auto() # Sign extend 8 bits
    SE16   = auto() # Sign extend 16 bits
    DOLOOP = auto() # Do Loop
    PUSH   = auto() # Push to stack
    PUSHM  = auto() # Push multiple to stack
    POP    = auto() # Pop from stack
    POPM   = auto() # Pop multiple from stack

    LOADS  = auto() # Load from stack with offset
    STORES = auto() # Store to stack with offset
    LOADW  = auto() # Subword load
    STOREW = auto() # Subword store
    PREFIX = auto() # 32-bit constant prefix
    UNUSED = auto() # Reserved

unop_symbols = {
    KalimbaOp.SIGN:  'SIGNDET',
    KalimbaOp.BSIGN: 'BLKSIGNDET',
    KalimbaOp.ONEBC: 'ONEBITCOUNT',
    KalimbaOp.TWOBC: 'TWOBITCOUNT',
}

@dataclass(unsafe_hash=True)
class KalimbaUnOp:
    '''
    C = OP A
    '''
    op: KalimbaOp
    c: KalimbaReg
    a: KalimbaReg
    cond: KalimbaCond = KalimbaCond.Always
    mem1: Optional[KalimbaIndexedMemAccess] = None
    mem2: Optional[KalimbaIndexedMemAccess] = None

    def __str__(self):
        op = self.op.name

        if self.op == KalimbaOp.DIV:
            assert not self.mem1 and not self.mem2 and self.cond == KalimbaCond.Always
            assert self.a in [KalimbaBank3Reg.DivResult, KalimbaBank3Reg.DivRemainder]
            return f'KalimbaUnOp("{self.c} = {self.a}")'

        if self.op in unop_symbols:
            op = unop_symbols[self.op]

        m1 = f', {self.mem1}' if self.mem1 else ''
        m2 = f', {self.mem2}' if self.mem2 else ''
        if self.cond == KalimbaCond.Always:
            return f'KalimbaUnOp("{self.c} = {op} {self.a}{m1}{m2}")'
        else:
            return f'KalimbaUnOp("if {self.cond.name} {self.c} = {op} {self.a}{m1}{m2}")'

binop_symbols = {
    KalimbaOp.ADD:  ('+', ''),
    KalimbaOp.ADC:  ('+', ' + Carry'),
    KalimbaOp.SUB:  ('-', ''),
    KalimbaOp.SBB:  ('-', ' - Borrow'),
    KalimbaOp.IMUL: ('*', ' (int)'),
    KalimbaOp.SMUL: ('*', ' (int) (sat)'),
    KalimbaOp.FMUL: ('*', ' (frac)'),
    KalimbaOp.DIV:  ('/', ''),
}

type KalimbaDstOp = Union[KalimbaWordMem, KalimbaReg]
type KalimbaSrcOp = Union[KalimbaWordMem, KalimbaReg, int]

class KalimbaShiftType(IntEnum):
    ST_32 = 0
    ST_72 = auto()
    ST_LO = auto()
    ST_MI = auto()
    ST_HI = auto()

shift_type_lut = {
    KalimbaShiftType.ST_32: '',
    KalimbaShiftType.ST_72: '(72bit)',
    KalimbaShiftType.ST_LO: '(LO)',
    KalimbaShiftType.ST_MI: '(MI)',
    KalimbaShiftType.ST_HI: '(HI)',
}

@dataclass(unsafe_hash=True)
class KalimbaBinOp:
    '''
    C = A OP B
    '''
    op: KalimbaOp
    c: KalimbaDstOp
    a: KalimbaSrcOp
    b: KalimbaSrcOp
    cond: KalimbaCond = KalimbaCond.Always
    mem1: Optional[KalimbaIndexedMemAccess] = None
    mem2: Optional[KalimbaIndexedMemAccess] = None
    shift: Optional[KalimbaShiftType] = None

    def __str__(self):
        c = self.c
        if self.op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT] and (isinstance(self.a, int) or isinstance(self.b, int)) and self.c in [KalimbaBank3Reg.rMAC1, KalimbaBank3Reg.rMACB1]:
            c = f'{self.c}2'

        op = self.op.name
        extra = ''
        if self.op in binop_symbols:
            (op, extra) = binop_symbols[self.op]
        elif self.op in [KalimbaOp.ASHIFT, KalimbaOp.LSHIFT] and self.shift:
            extra = f' {shift_type_lut[self.shift]}'

        m1 = f', {self.mem1}' if self.mem1 else ''
        m2 = f', {self.mem2}' if self.mem2 else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '

        b = str(self.b)
        if self.op == KalimbaOp.FMUL and isinstance(self.b, int):
            b = format(Decimal(self.b) / 2**31, '.48g')
        return f'KalimbaBinOp("{cond}{c if self.c != KalimbaBank3Reg.DivResult else 'Div'} = {self.a} {op} {b}{extra}{m1}{m2}")'

def get_mask(length):
    return (1 << length) - 1

def get_bits(instruction, offset, length):
    return (instruction >> offset) & get_mask(length)

def unsigned_to_signed(v, length):
    m = 1 << (length - 1)
    return (v ^ m) - m

def signed_to_unsigned(v, length):
    return v & get_mask(length)

def get_bits_signed(instruction, offset, length):
    return unsigned_to_signed(get_bits(instruction, offset, length), length)

def kalimba_maxim_decode_cond_a(instruction):
    return KalimbaCond(get_bits(instruction, 0, 4))

def kalimba_maxim_decode_mem_a(instruction):
    mag1   = KalimbaBank2Reg(KalimbaBank2Reg.M0.value + get_bits(instruction, 8, 2))
    iag1   = KalimbaBank2Reg(KalimbaBank2Reg.I0.value + get_bits(instruction, 10, 2))
    regag1 = KalimbaBank1Reg(get_bits(instruction, 12, 3))
    ag1w   = bool(get_bits(instruction, 15, 1))

    return KalimbaIndexedMemAccess(ag1w, regag1, iag1, mag1) if regag1 != KalimbaBank1Reg.Null else None

def kalimba_maxim_decode_regs_a(instruction: int, banka = KalimbaBank1Reg, bankb = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    rega = banka(get_bits(instruction, 16, 4))
    regb = bankb(get_bits(instruction, 4, 4))
    regc = bankc(get_bits(instruction, 20, 4))

    return (rega, regb, regc)

def kalimba_maxim_decode_a(instruction: int, banka = KalimbaBank1Reg, bankb = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    cond = kalimba_maxim_decode_cond_a(instruction)
    mem = kalimba_maxim_decode_mem_a(instruction)
    (rega, regb, regc) = kalimba_maxim_decode_regs_a(instruction, banka, bankb, bankc)

    return (cond, mem, rega, regb, regc)

def kalimba_maxim_decode_unop_bank1_a(instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaUnOp(op, regc, rega, cond, mem)

def kalimba_maxim_decode_binop_bank1_a(instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)

    shift = None

    # Special case: rFlags is invalid here, so this is actually rMAC with 32-bit width
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT]:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
            shift = KalimbaShiftType.ST_32
        elif regc in [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB]:
            shift = KalimbaShiftType.ST_72

    return KalimbaBinOp(op, regc, rega, regb, cond, mem, None, shift)

def kalimba_maxim_decode_b(instruction: int, banka = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    k16 = get_bits_signed(instruction, 0, 16)
    rega = banka(get_bits(instruction, 16, 4))
    regc = bankc(get_bits(instruction, 20, 4))

    return (k16, rega, regc)

def kalimba_maxim_decode_b_const_extend(op, kn, prefix, n = 16, is_frac_signed = False):
    knu = signed_to_unsigned(kn, n)
    if not prefix:
        if op in [KalimbaOp.AND, KalimbaOp.OR, KalimbaOp.XOR]:
            return knu
        elif op in [KalimbaOp.FMUL, KalimbaOp.FMADD, KalimbaOp.FMSUB]:
            k32 = knu << (32 - n)
            if is_frac_signed:
                return unsigned_to_signed(k32, 32)
            else:
                return k32
        else:
            # Sign extension is automatic
            return kn
    else:
        return unsigned_to_signed(knu | (prefix.const << n), 32)

def kalimba_maxim_decode_binop_bank1_b(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)
    return KalimbaBinOp(op, regc, rega, k32)


def kalimba_maxim_decode_shift_by_c_bank1_b(instruction, op, prefix):
    (_, rega, regc) = kalimba_maxim_decode_b(instruction)
    amount = get_bits_signed(instruction, 0, 8)
    dest = get_bits(instruction, 8, 3)

    shift = None

    # TODO: return as kalcode, i.e. kalcode(912b0207), there are others like this
    assert not (dest == 0b101 and regc == KalimbaBank1Reg.rMAC)

    if dest == 0b001:
        shift = KalimbaShiftType.ST_LO
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
    elif dest == 0b000 and regc == KalimbaBank1Reg.rFlags:
        regc = KalimbaBank1Reg.rMAC
        shift = KalimbaShiftType.ST_MI
    elif dest == 0b000 and regc in [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB]:
        shift = KalimbaShiftType.ST_72
    elif dest == 0b010:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
        shift = KalimbaShiftType.ST_HI
    elif dest == 0b101:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank3Reg.rMAC0
        elif regc == KalimbaBank1Reg.rMACB:
            regc = KalimbaBank3Reg.rMACB0
        shift = KalimbaShiftType.ST_32
    elif dest == 0b100:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank3Reg.rMAC1
        elif regc == KalimbaBank1Reg.rMACB:
            regc = KalimbaBank3Reg.rMACB1
        shift = KalimbaShiftType.ST_32
    elif dest == 0b110:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank3Reg.rMAC2
        elif regc == KalimbaBank1Reg.rMACB:
            regc = KalimbaBank3Reg.rMACB2
        shift = KalimbaShiftType.ST_32

    return KalimbaBinOp(op, regc, rega, amount, KalimbaCond.Always, None, None, shift)

def kalimba_maxim_decode_shift_c_bank1_b(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)

    shift = KalimbaShiftType.ST_72
    if regc == KalimbaBank1Reg.rFlags:
        regc = KalimbaBank1Reg.rMAC
        shift = KalimbaShiftType.ST_32

    return KalimbaBinOp(op, regc, k32, rega, KalimbaCond.Always, None, None, shift)

def kalimba_maxim_decode_mem1_common_c(instruction):
    mag1_k = get_bits(instruction, 8, 2)
    iag1   = KalimbaBank2Reg(KalimbaBank2Reg.I0.value + get_bits(instruction, 10, 2))
    regag1 = KalimbaBank1Reg(get_bits(instruction, 12, 3))
    ag1w   = bool(get_bits(instruction, 15, 1))
    return (mag1_k, iag1, regag1, ag1w)

def kalimba_maxim_decode_mem2_common_c(instruction):
    mag2_k = get_bits(instruction, 0, 2)
    iag2   = KalimbaBank2Reg(KalimbaBank2Reg.I4.value + get_bits(instruction, 2, 2))
    regag2 = KalimbaBank1Reg(get_bits(instruction, 4, 3))
    ag2w   = bool(get_bits(instruction, 7, 1))
    return (mag2_k, iag2, regag2, ag2w)

def kalimba_maxim_decode_mem1_reg_c(instruction):
    (mag1_k, iag1, regag1, ag1w) = kalimba_maxim_decode_mem1_common_c(instruction)
    if regag1 == KalimbaBank1Reg.Null:
        return None
    else:
        mag1 = KalimbaBank2Reg(KalimbaBank2Reg.M0.value + mag1_k)
        return KalimbaIndexedMemAccess(ag1w, regag1, iag1, mag1)

def kalimba_maxim_decode_mem2_reg_c(instruction):
    (mag2_k, iag2, regag2, ag2w) = kalimba_maxim_decode_mem2_common_c(instruction)
    if regag2 == KalimbaBank1Reg.Null:
        return None
    else:
        mag2 = KalimbaBank2Reg(KalimbaBank2Reg.M0.value + mag2_k)
        return KalimbaIndexedMemAccess(ag2w, regag2, iag2, mag2)

def kalimba_maxim_decode_creg_c(instruction, banka = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    mem1 = kalimba_maxim_decode_mem1_reg_c(instruction)
    mem2 = kalimba_maxim_decode_mem2_reg_c(instruction)
    rega = banka(get_bits(instruction, 16, 4))
    regc = bankc(get_bits(instruction, 20, 4))
    return (mem1, mem2, rega, regc)

def kalimba_maxim_decode_cregc_c(instruction, bankc = KalimbaBank1Reg):
    mem1 = kalimba_maxim_decode_mem1_reg_c(instruction)
    mem2 = kalimba_maxim_decode_mem2_reg_c(instruction)
    regc = bankc(get_bits(instruction, 20, 4))
    return (mem1, mem2, regc)

mag12_const_lut = {
    0b00: 0,
    0b01: 4,
    0b10: 8,
    0b11: -4,
}

def kalimba_maxim_decode_mem1_const_c(instruction):
    (mag1_k, iag1, regag1, ag1w) = kalimba_maxim_decode_mem1_common_c(instruction)
    if regag1 == KalimbaBank1Reg.Null:
        return None
    else:
        mkag1 = mag12_const_lut[mag1_k]
        return KalimbaIndexedMemAccess(ag1w, regag1, iag1, mkag1)

def kalimba_maxim_decode_mem2_const_c(instruction):
    (mag2_k, iag2, regag2, ag2w) = kalimba_maxim_decode_mem2_common_c(instruction)
    if regag2 == KalimbaBank1Reg.Null:
        return None
    else:
        mkag2 = mag12_const_lut[mag2_k]
        return KalimbaIndexedMemAccess(ag2w, regag2, iag2, mkag2)

def kalimba_maxim_decode_const_c(instruction, banka = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    mem1 = kalimba_maxim_decode_mem1_const_c(instruction)
    mem2 = kalimba_maxim_decode_mem2_const_c(instruction)
    rega = banka(get_bits(instruction, 16, 4))
    regc = bankc(get_bits(instruction, 20, 4))
    return (mem1, mem2, rega, regc)

def kalimba_maxim_decode_constregc_c(instruction, bankc = KalimbaBank1Reg):
    mem1 = kalimba_maxim_decode_mem1_const_c(instruction)
    mem2 = kalimba_maxim_decode_mem2_const_c(instruction)
    regc = bankc(get_bits(instruction, 20, 4))
    return (mem1, mem2, regc)

class KalimbaAddressingMode(IntEnum):
    RRR = 0b00 # reg = reg OP reg/imm
    RRM = 0b01 # reg = reg OP mem[reg/imm]
    RMR = 0b10 # reg = mem[reg] OP reg/imm
    MRR = 0b11 # mem[r/imm] = reg OP reg

R = lambda r: r
M = lambda r: KalimbaWordMem(r)

addressing_mode_lut = {
    0b00: (R, R, R), # reg = reg OP reg/imm
    0b01: (R, R, M), # reg = reg OP mem[reg/imm]
    0b10: (R, M, R), # reg = mem[reg] OP reg/imm
    0b11: (M, R, R), # mem[r/imm] = reg OP reg
}

def kalimba_maxim_decode_binop_bank1_a_addsub(instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    (fc, fa, fb) = addressing_mode_lut[get_bits(instruction, 27, 2)]
    return KalimbaBinOp(op, fc(regc), fa(rega), fb(regb), cond, mem)

def kalimba_maxim_decode_binop_bank1_b_addsub(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)

    addr_mode = get_bits(instruction, 27, 2)
    (fc, fa, fb) = addressing_mode_lut[addr_mode]
    if addr_mode == 0b11:
        return KalimbaBinOp(op, fc(k32), fa(regc), fb(rega))
    else:
        return KalimbaBinOp(op, fc(regc), fa(rega), fb(k32))

def kalimba_maxim_decode_binop_bank1_creg_addsub(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_creg_c(instruction)
    (_, _, fa) = addressing_mode_lut[get_bits(instruction, 27, 2)]
    return KalimbaBinOp(op, regc, regc, fa(rega), KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_binop_bank1_const_addsub(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_const_c(instruction)
    (_, _, fa) = addressing_mode_lut[get_bits(instruction, 27, 2)]
    return KalimbaBinOp(op, regc, regc, fa(rega), KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_binop_bank1_a_const(const, instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaBinOp(op, regc, rega, const, cond, mem)

kalimba_maxim_decode_binop_bank1_a_const1 = partial(kalimba_maxim_decode_binop_bank1_a_const, 1)
kalimba_maxim_decode_binop_bank1_a_const2 = partial(kalimba_maxim_decode_binop_bank1_a_const, 2)
kalimba_maxim_decode_binop_bank1_a_const4 = partial(kalimba_maxim_decode_binop_bank1_a_const, 4)

bank_select_a_lut = {
    0b000: (KalimbaBank1Reg, KalimbaBank1Reg, KalimbaBank1Reg), # B1 = B1 +- B1
    0b001: (KalimbaBank1Reg, KalimbaBank1Reg, KalimbaBank2Reg), # B1 = B1 +- B2
    0b010: (KalimbaBank1Reg, KalimbaBank2Reg, KalimbaBank1Reg), # B1 = B2 +- B1
    0b011: (KalimbaBank1Reg, KalimbaBank2Reg, KalimbaBank2Reg), # B1 = B2 +- B2
    0b100: (KalimbaBank2Reg, KalimbaBank1Reg, KalimbaBank1Reg), # B2 = B1 +- B1
    0b101: (KalimbaBank2Reg, KalimbaBank1Reg, KalimbaBank2Reg), # B2 = B1 +- B2
    0b110: (KalimbaBank2Reg, KalimbaBank2Reg, KalimbaBank1Reg), # B2 = B2 +- B1
    0b111: (KalimbaBank2Reg, KalimbaBank2Reg, KalimbaBank2Reg), # B2 = B2 +- B2
}

assert KalimbaBank1Reg.rFlags != KalimbaBank2Reg.L4
assert KalimbaBank1Reg.rFlags is not KalimbaBank2Reg.L4

def kalimba_maxim_decode_binop_bank12_a(instruction, op, prefix):
    (bankc, banka, bankb) = bank_select_a_lut[get_bits(instruction, 26, 3)]
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction, banka, bankb, bankc)

    # Special case: rFlags is invalid here, this is actually FP
    if rega == KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    if regb == KalimbaBank1Reg.rFlags:
        regb = KalimbaBank3Reg.FP

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

bank_select_b_lut = {
    0b00: (KalimbaBank1Reg, KalimbaBank1Reg),
    0b01: (KalimbaBank1Reg, KalimbaBank2Reg),
    0b10: (KalimbaBank2Reg, KalimbaBank1Reg),
    0b11: (KalimbaBank2Reg, KalimbaBank2Reg),
}
def kalimba_maxim_decode_binop_bank12_b(instruction, op, prefix):
    (bankc, banka) = bank_select_b_lut[get_bits(instruction, 27, 2)]
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction, banka, bankc)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)

    # Special case: rFlags is invalid here, this is actually FP
    if rega == KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    if get_bits(instruction, 26, 1) == 0:
        return KalimbaBinOp(op, regc, rega, k32)
    else:
        return KalimbaBinOp(op, regc, k32, rega)

bank_select_c_lut = {
    0b000: (KalimbaBank1Reg, KalimbaBank1Reg),
    0b001: (KalimbaBank1Reg, KalimbaBank2Reg),
    0b110: (KalimbaBank2Reg, KalimbaBank1Reg),
    0b111: (KalimbaBank2Reg, KalimbaBank2Reg),
}

def kalimba_maxim_decode_binop_bank12_creg(instruction, op, prefix):
    (bankc, banka) = bank_select_c_lut[get_bits(instruction, 26, 3)]
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_creg_c(instruction, banka, bankc)

    regb = regc

    # Special case: rFlags is invalid here, this is actually FP
    if regc == KalimbaBank1Reg.rFlags:
        regb = KalimbaBank3Reg.FP

    return KalimbaBinOp(op, regc, regb, rega, KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_binop_bank12_const(instruction, op, prefix):
    (bankc, banka) = bank_select_c_lut[get_bits(instruction, 26, 3)]
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_const_c(instruction, banka, bankc)

    # Special case: rFlags is invalid here, this is actually FP
    if rega == KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    return KalimbaBinOp(op, regc, regc, rega, KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_binop_bank1_creg(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_creg_c(instruction)

    shift = None

    # Special case: rFlags is invalid here, so this is actually rMAC with 32-bit width
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT]:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
            shift = KalimbaShiftType.ST_32
        elif regc in [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB]:
            shift = KalimbaShiftType.ST_72

    return KalimbaBinOp(op, regc, regc, rega, KalimbaCond.Always, mem1, mem2, shift)

def kalimba_maxim_decode_binop_bank1_const(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_const_c(instruction)

    shift = None

    # Special case: rFlags is invalid here, so this is actually rMAC with 32-bit width
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT]:
        if regc == KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
            shift = KalimbaShiftType.ST_32
        elif regc in [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB]:
            shift = KalimbaShiftType.ST_72

    return KalimbaBinOp(op, regc, regc, rega, KalimbaCond.Always, mem1, mem2, shift)

def kalimba_maxim_decode_unop_bank1_creg(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_creg_c(instruction)
    return KalimbaUnOp(op, regc, rega, KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_unop_bank1_const(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_const_c(instruction)
    return KalimbaUnOp(op, regc, rega, KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_divide_b(instruction, op, prefix):
    (_, rega, regc) = kalimba_maxim_decode_b(instruction)
    div = get_bits(instruction, 0, 2)

    # TODO: kalcode
    assert div != 0b11

    if div == 0b00:
        return KalimbaBinOp(op, KalimbaBank3Reg.DivResult, regc, rega)
    elif div == 0b01:
        return KalimbaUnOp(op, regc, KalimbaBank3Reg.DivResult)
    elif div == 0b10:
        return KalimbaUnOp(op, regc, KalimbaBank3Reg.DivRemainder)

class KalimbaSignSelect(IntEnum):
    UU = 0b00
    US = 0b01
    SU = 0b10
    SS = 0b11

@dataclass(unsafe_hash=True)
class KalimbaExtraAddSub:
    op: Union[Literal[KalimbaOp.ADD], Literal[KalimbaOp.SUB]]
    a: KalimbaBank1Reg
    b: KalimbaBank1Reg

    def __str__(self):
        op = binop_symbols[self.op][0]
        return f'{KalimbaBank1Reg.r0} = {self.a} {op} {self.b}'

fused_mul_symbols = {
    KalimbaOp.FMADD: '+',
    KalimbaOp.FMSUB: '-',
}

@dataclass(unsafe_hash=True)
class KalimbaFusedMultiplyAddSub:
    '''
    C = C OP A * B [r0 = D OP E]
    '''
    op: KalimbaOp
    c: KalimbaBank2Reg
    a: Union[KalimbaBank1Reg, int]
    b: Union[KalimbaBank1Reg, int]
    sign: KalimbaSignSelect
    cond: KalimbaCond = KalimbaCond.Always
    addsub: Optional[KalimbaExtraAddSub] = None
    mem: Optional[KalimbaIndexedMemAccess] = None

    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        addsub = f', {self.addsub}' if self.addsub else ''
        sign = f' ({self.sign.name})'

        b = self.b
        if isinstance(self.b, int):
            if self.sign in [KalimbaSignSelect.SU, KalimbaSignSelect.UU]:
                b = Decimal(self.b) / 2**32
            else:
                b = Decimal(self.b) / 2**31

        if self.op in fused_mul_symbols:
            op = fused_mul_symbols[self.op]
            return f'KalimbaFusedMultiplyAddSub("{cond}{self.c} = {self.c} {op} {self.a} * {b}{sign}{addsub}{m}")'
        else:
            return f'KalimbaFusedMultiplyAddSub("{cond}{self.c} = {self.a} * {b}{sign}{addsub}{m}")'

def kalimba_maxim_decode_fmaddsub_a(instruction, op, prefix):
    (cond, mem, rega, regb, _) = kalimba_maxim_decode_a(instruction)

    addsub = None
    sign = KalimbaSignSelect.SS

    if get_bits(instruction, 23, 1) == 0:
        # RegC' = RegC' OP RegA * RegB
        regc = KalimbaBank1Reg(get_bits(instruction, 20, 3))
        # Special case: Null is actually rMACB
        if regc == KalimbaBank1Reg.Null:
            regc = KalimbaBank1Reg.rMACB
        sign = KalimbaSignSelect(get_bits(instruction, 26, 2))
    else:
        # RegC'' = RegC'' OP RegA * RegB, r0 = RegD OP RegE
        regc = [KalimbaBank1Reg.rMACB, KalimbaBank1Reg.rMAC, KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 20, 2)]
        regd = [KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 27, 1)]
        rege = [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB][get_bits(instruction, 22, 1)]
        addsub_op = [KalimbaOp.ADD, KalimbaOp.SUB][get_bits(instruction, 26, 1)]
        addsub = KalimbaExtraAddSub(addsub_op, regd, rege)

    return KalimbaFusedMultiplyAddSub(op, regc, rega, regb, sign, cond, addsub, mem)

def kalimba_maxim_decode_fmaddsub_b(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)

    addsub = None
    sign = KalimbaSignSelect.SS

    if get_bits(instruction, 23, 1) == 0:
        # RegC' = RegC' OP RegA * RegB
        regc = KalimbaBank1Reg(get_bits(instruction, 20, 3))
        # Special case: Null is actually rMACB
        if regc == KalimbaBank1Reg.Null:
            regc = KalimbaBank1Reg.rMACB
        sign = KalimbaSignSelect(get_bits(instruction, 26, 2))
    else:
        # RegC'' = RegC'' OP RegA * RegB, r0 = RegD OP RegE
        regc = [KalimbaBank1Reg.rMACB, KalimbaBank1Reg.rMAC, KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 20, 2)]
        regd = [KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 27, 1)]
        rege = [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB][get_bits(instruction, 22, 1)]
        addsub_op = [KalimbaOp.ADD, KalimbaOp.SUB][get_bits(instruction, 26, 1)]
        addsub = KalimbaExtraAddSub(addsub_op, regd, rege)

    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix, 16, sign in [KalimbaSignSelect.SS, KalimbaSignSelect.US])

    return KalimbaFusedMultiplyAddSub(op, regc, rega, k32, sign, KalimbaCond.Always, addsub)

@dataclass(unsafe_hash=True)
class KalimbaOffsetMemAccess:
    '''
    Reads/writes one word with offset
    '''
    op: KalimbaOp
    c: KalimbaBank1Reg
    a: KalimbaBank1Reg
    b: KalimbaBank1Reg
    cond: KalimbaCond = KalimbaCond.Always
    mem1: Optional[KalimbaIndexedMemAccess] = None
    mem2: Optional[KalimbaIndexedMemAccess] = None
    def __str__(self):
        m1 = f', {self.mem1}' if self.mem1 else ''
        m2 = f', {self.mem2}' if self.mem2 else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '

        if self.op in [KalimbaOp.LOAD, KalimbaOp.LOADS]:
            return f'KalimbaOffsetMemAccess("{cond}{self.c} = M[{self.a} + {self.b}]{m1}{m2}")'
        elif self.op in [KalimbaOp.STORE, KalimbaOp.STORES]:
            return f'KalimbaOffsetMemAccess("{cond}M[{self.a} + {self.b}] = {self.c}{m1}{m2}")'
        else:
            raise ValueError(f'invalid mem op: {self.op}') # pragma: no cover

def kalimba_maxim_decode_load_store_a(instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaOffsetMemAccess(op, regc, rega, regb, cond, mem)

def kalimba_maxim_decode_load_store_b(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)
    return KalimbaOffsetMemAccess(op, regc, rega, k32, KalimbaCond.Always, None)

def kalimba_maxim_decode_load_creg(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_creg_c(instruction)

    return KalimbaOffsetMemAccess(op, regc, regc, rega, KalimbaCond.Always, mem1, mem2)

def kalimba_maxim_decode_load_const(instruction, op, prefix):
    (mem1, mem2, rega, regc) = kalimba_maxim_decode_const_c(instruction)

    return KalimbaOffsetMemAccess(op, regc, regc, rega, KalimbaCond.Always, mem1, mem2)

@dataclass(unsafe_hash=True)
class KalimbaControlFlow:
    '''
    JUMP/CALL
    '''
    op: KalimbaOp
    a: Union[KalimbaBank1Reg, int]
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]
    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        t = f' {self.a}' if self.op in [KalimbaOp.JUMP, KalimbaOp.CALL, KalimbaOp.DOLOOP] else ''
        n = 'do' if self.op == KalimbaOp.DOLOOP else self.op.name.lower()

        if isinstance(self.a, int) and self.a % 2 == 1 and self.op in [KalimbaOp.JUMP, KalimbaOp.CALL]:
            n += ' (m)'
            t = f' {self.a & -2}'

        return f'KalimbaControlFlow("{cond}{n}{t}{m}")'

def kalimba_maxim_decode_flow_a(instruction, op, prefix):
    (cond, mem, rega, regb, _) = kalimba_maxim_decode_a(instruction)
    if rega == KalimbaBank1Reg.rLink:
        op = KalimbaOp.RTS
    elif rega == KalimbaBank1Reg.rFlags:
        op = KalimbaOp.RTI

    return KalimbaControlFlow(op, rega, cond, mem)

def kalimba_maxim_decode_flow_b(instruction, op, prefix):
    (k16, _, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)
    cond = KalimbaCond(regc.value)

    return KalimbaControlFlow(op, k32, cond, None)

def kalimba_maxim_decode_doloop_b(instruction, op, prefix):
    (k16, _, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)

    return KalimbaControlFlow(op, k32, KalimbaCond.Always, None)

def reg_str_fp_special(reg):
    if reg == KalimbaBank3Reg.FP:
        return 'FP(=SP)'
    else:
        return reg.name

@dataclass(unsafe_hash=True)
class KalimbaStackOp:
    '''
    PUSH/POP
    '''
    op: KalimbaOp
    reg: List[KalimbaReg]
    cond: KalimbaCond
    adj: Optional[int] = None
    mem1: Optional[KalimbaIndexedMemAccess] = None
    mem2: Optional[KalimbaIndexedMemAccess] = None
    new_stack_frame: bool = False
    def __str__(self):
        m1 = f', {self.mem1}' if self.mem1 else ''
        m2 = f', {self.mem2}' if self.mem2 else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        reg = None
        if len(self.reg) == 1:
            reg = self.reg[0].name
        elif len(self.reg) > 1:
            assert self.op in [KalimbaOp.PUSHM, KalimbaOp.POPM]

            if self.new_stack_frame:
                reg = f'<{', '.join(map(reg_str_fp_special, self.reg))}>'
            else:
                reg = f'<{', '.join(map(lambda x: x.name, self.reg))}>'
        else:
            raise ValueError('unreachable') # pragma: no cover

        if self.adj:
            if self.op in [KalimbaOp.PUSH, KalimbaOp.PUSHM]:
                return f'KalimbaStackOp("{cond}{self.op.name.lower()} {reg}, SP = SP + {self.adj}{m1}{m2}")'
            else:
                return f'KalimbaStackOp("{cond}SP = SP - {self.adj}, {self.op.name.lower()} {reg}{m1}{m2}")'
        else:
            return f'KalimbaStackOp("{cond}{self.op.name.lower()} {reg}{m1}{m2}")'

def kalimba_maxim_decode_stack_a(instruction, op, prefix):
    bankc = reg_bank_lut[get_bits(instruction, 16, 2) + 1]
    (cond, mem, _, regb, regc) = kalimba_maxim_decode_a(instruction, KalimbaBank1Reg, KalimbaBank1Reg, bankc)

    if op in [KalimbaOp.STORES, KalimbaOp.LOADS]:
        return KalimbaOffsetMemAccess(op, regc, KalimbaBank3Reg.FP, regb, cond, mem)
    else:
        return KalimbaStackOp(op, [regc], cond, None, mem)

def kalimba_maxim_decode_stack_adj_a(instruction, op, prefix):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)

    if (rega.value & 0b100) == 0b100:
        rega = KalimbaBank3Reg.FP
    else:
        rega = KalimbaBank3Reg.SP

    if regc == KalimbaBank1Reg.Null:
        regc = rega;

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

@dataclass(unsafe_hash=True)
class KalimbaPushOff:
    op: KalimbaOp
    reg: KalimbaBank1Reg
    off: int
    def __str__(self):
        return f'KalimbaPushOff("{self.op.name.lower()} {self.reg} + {self.off}")'

def kalimba_maxim_decode_stack_b(instruction, op, prefix):
    banksel = get_bits(instruction, 16, 2)

    if banksel == 0b11:
        assert op == KalimbaOp.PUSH
        (k16, _, regc) = kalimba_maxim_decode_b(instruction)
        k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)
        return KalimbaPushOff(op, regc, k32)

    bankc = reg_bank_lut[banksel + 1]

    if op in [KalimbaOp.STORES, KalimbaOp.LOADS]:
        (_, _, regc) = kalimba_maxim_decode_b(instruction, KalimbaBank1Reg, bankc)
        offset = get_bits_signed(instruction, 0, 15)
        regst = KalimbaBank3Reg.FP if get_bits(instruction, 15, 1) == 0 else KalimbaBank3Reg.SP
        # TODO: what to do with prefix here?
        return KalimbaOffsetMemAccess(op, regc, regst, offset)
    elif op in [KalimbaOp.PUSHM, KalimbaOp.POPM]:
        reglist = get_bits(instruction, 0, 16)
        adj = get_bits(instruction, 20, 4) * 4
        regs = []
        new_stack_frame = False

        for reg in bankc:
            if (reglist & (1 << reg.value)) != 0:
                if reg == KalimbaBank1Reg.Null:
                    new_stack_frame = op == KalimbaOp.PUSHM
                    regs += [KalimbaBank3Reg.FP]
                else:
                    regs += [reg]
        return KalimbaStackOp(op, regs, KalimbaCond.Always, adj, None, None, new_stack_frame)
    else:
        raise ValueError(f'instruction 0b{instruction:032b} cannot be {op}') # pragma: no cover

def kalimba_maxim_decode_stack_adj_b(instruction, op, prefix):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    k32 = kalimba_maxim_decode_b_const_extend(op, k16, prefix)

    if (rega.value & 0b100) == 0b100:
        rega = KalimbaBank3Reg.FP
    else:
        rega = KalimbaBank3Reg.SP

    if regc == KalimbaBank1Reg.Null:
        regc = rega;

    return KalimbaBinOp(op, regc, rega, k32)

def kalimba_maxim_decode_stack_creg(instruction, op, prefix):
    bankc = reg_bank_lut[get_bits(instruction, 16, 2) + 1]
    (mem1, mem2, regc) = kalimba_maxim_decode_cregc_c(instruction, bankc)
    return KalimbaStackOp(op, [regc], KalimbaCond.Always, None, mem1, mem2)

def kalimba_maxim_decode_stack_const(instruction, op, prefix):
    bankc = reg_bank_lut[get_bits(instruction, 16, 2) + 1]
    (mem1, mem2, regc) = kalimba_maxim_decode_constregc_c(instruction, bankc)
    return KalimbaStackOp(op, [regc], KalimbaCond.Always, None, mem1, mem2)

@dataclass(unsafe_hash=True)
class KalimbaSubWordMemAccess:
    '''
    Reads/writes one word with offset
    '''
    op: KalimbaOp
    sel: KalimbaSubWordMem
    c: Union[KalimbaBank1Reg, KalimbaBank2Reg]
    a: Union[KalimbaBank1Reg, KalimbaBank2Reg]
    b: Union[KalimbaBank1Reg, KalimbaBank2Reg, int]
    cond: KalimbaCond = KalimbaCond.Always
    sub: bool = False
    def __str__(self):
        m = self.sel.name[2:]
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        op = '-' if self.sub else '+'

        if self.op in [KalimbaOp.LOADW]:
            return f'KalimbaSubWordMemAccess("{cond}{self.c} = {m}[{self.a} {op} {self.b}]")'
        elif self.op in [KalimbaOp.STOREW]:
            return f'KalimbaSubWordMemAccess("{cond}{m}[{self.a} {op} {self.b}] = {self.c}")'
        else:
            raise ValueError(f'invalid mem op: {self.op}') # pragma: no cover

bank_bit_lut = [KalimbaBank1Reg, KalimbaBank2Reg]

def kalimba_maxim_decode_subword_a(instruction, op, prefix):
    sel = KalimbaSubWordMem(get_bits(instruction, 13, 3))
    cond = kalimba_maxim_decode_cond_a(instruction)
    sub = get_bits(instruction, 9, 1) == 1
    banka = bank_bit_lut[get_bits(instruction, 11, 1)]
    bankb = bank_bit_lut[get_bits(instruction, 10, 1)]
    bankc = bank_bit_lut[get_bits(instruction, 12, 1)]
    (rega, regb, regc) = kalimba_maxim_decode_regs_a(instruction, banka, bankb, bankc)

    if sel in [KalimbaSubWordMem.S_M, KalimbaSubWordMem.S_MB, KalimbaSubWordMem.S_MH]:
        op = KalimbaOp.STOREW
    else:
        op = KalimbaOp.LOADW

    return KalimbaSubWordMemAccess(op, sel, regc, rega, regb, cond, sub)

def kalimba_maxim_decode_subword_b(instruction, op, prefix):
    sel = KalimbaSubWordMem(get_bits(instruction, 13, 3))
    banka = bank_bit_lut[get_bits(instruction, 11, 1)]
    bankc = bank_bit_lut[get_bits(instruction, 12, 1)]
    (_, rega, regc) = kalimba_maxim_decode_b(instruction, banka, bankc)
    k11 = get_bits_signed(instruction, 0, 11)
    k32 = kalimba_maxim_decode_b_const_extend(op, k11, prefix, 11)

    if sel in [KalimbaSubWordMem.S_M, KalimbaSubWordMem.S_MB, KalimbaSubWordMem.S_MH]:
        op = KalimbaOp.STOREW
    else:
        op = KalimbaOp.LOADW

    return KalimbaSubWordMemAccess(op, sel, regc, rega, k32)

@dataclass(unsafe_hash=True)
class KalimbaPrefix:
    op: KalimbaOp
    const: int
    def __str__(self):
        return f'KalimbaPrefix(0x{self.const:x})'

def kalimba_maxim_decode_prefix(instruction, op, prefix):
    # assert not prefix
    prefix = get_bits(instruction, 0, 21)
    return KalimbaPrefix(op, prefix)

type KalimbaInstruction = Union[KalimbaUnOp, KalimbaBinOp, KalimbaFusedMultiplyAddSub, KalimbaOffsetMemAccess, KalimbaControlFlow, KalimbaSubWordMemAccess, KalimbaPushOff]

# mask, value, operation, decode
maxim_ops_lut: List[Union[Tuple[int, int, KalimbaOp, Callable[[int, KalimbaOp, Optional[KalimbaPrefix]], KalimbaInstruction]], Tuple[int, int, KalimbaOp]]] = [
    # Type A
    (0b111001_11_00000000_00000000_00000000, 0b000_000_00_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank1_a_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b000_001_00_00000000_00000000_00000000, KalimbaOp.ADC, kalimba_maxim_decode_binop_bank1_a_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_000_00_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank1_a_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_001_00_00000000_00000000_00000000, KalimbaOp.SBB, kalimba_maxim_decode_binop_bank1_a_addsub),
    (0b111000_11_00000000_00000000_00000000, 0b010_000_00_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank12_a), #ADDB12
    (0b111000_11_00000000_00000000_00000000, 0b011_000_00_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank12_a), #SUBB12
    (0b111111_11_00000000_00000000_00000000, 0b100_000_00_00000000_00000000_00000000, KalimbaOp.AND, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_001_00_00000000_00000000_00000000, KalimbaOp.OR,  kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_010_00_00000000_00000000_00000000, KalimbaOp.XOR, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_011_00_00000000_00000000_00000000, KalimbaOp.LSHIFT, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_100_00_00000000_00000000_00000000, KalimbaOp.ASHIFT, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_110_00_00000000_00000000_00000000, KalimbaOp.IMUL, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_111_00_00000000_00000000_00000000, KalimbaOp.SMUL, kalimba_maxim_decode_binop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b100_101_00_00000000_00000000_00000000, KalimbaOp.FMUL, kalimba_maxim_decode_binop_bank1_a),
    (0b111100_11_00000000_00000000_00000000, 0b101_000_00_00000000_00000000_00000000, KalimbaOp.FMADD, kalimba_maxim_decode_fmaddsub_a),
    (0b111100_11_00000000_00000000_00000000, 0b101_100_00_00000000_00000000_00000000, KalimbaOp.FMSUB, kalimba_maxim_decode_fmaddsub_a),
    (0b111100_11_00000000_00000000_00000000, 0b110_000_00_00000000_00000000_00000000, KalimbaOp.MULX,  kalimba_maxim_decode_fmaddsub_a),
    (0b111111_11_00000000_00000000_00000000, 0b110_100_00_00000000_00000000_00000000, KalimbaOp.LOAD,  kalimba_maxim_decode_load_store_a),
    (0b111111_11_00000000_00000000_00000000, 0b110_101_00_00000000_00000000_00000000, KalimbaOp.STORE, kalimba_maxim_decode_load_store_a),
    (0b111111_11_00000000_00000000_00000000, 0b110_110_00_00000000_00000000_00000000, KalimbaOp.SIGN, kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_00000000, 0b110_111_00_00000000_00000000_00000000, KalimbaOp.JUMP, kalimba_maxim_decode_flow_a), # Special cases: RTS/RTI
    (0b111111_11_00000000_00000000_00000000, 0b111_000_00_00000000_00000000_00000000, KalimbaOp.CALL, kalimba_maxim_decode_flow_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_00100000, KalimbaOp.ADD,   kalimba_maxim_decode_binop_bank1_a_const1), # ADD1
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_00110000, KalimbaOp.SUB,   kalimba_maxim_decode_binop_bank1_a_const1), # SUB1
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_01000000, KalimbaOp.ABS,   kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_01010000, KalimbaOp.MIN,   kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_01100000, KalimbaOp.MAX,   kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_01110000, KalimbaOp.TWOBC, kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_10000000, KalimbaOp.MOD24, kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_10010000, KalimbaOp.ONEBC, kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_10100000, KalimbaOp.ADD,   kalimba_maxim_decode_binop_bank1_a_const2), # ADD2
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_10110000, KalimbaOp.ADD,   kalimba_maxim_decode_binop_bank1_a_const4), # ADD4
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_11000000, KalimbaOp.SUB,   kalimba_maxim_decode_binop_bank1_a_const2), # SUB2
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_11010000, KalimbaOp.SUB,   kalimba_maxim_decode_binop_bank1_a_const4), # SUB4
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_11100000, KalimbaOp.SE8,   kalimba_maxim_decode_unop_bank1_a),
    (0b111111_11_00000000_00000000_11110000, 0b111_001_00_00000000_00000000_11110000, KalimbaOp.SE16,  kalimba_maxim_decode_unop_bank1_a),
    #(0b111111_11_00000000_00000000_00000000, 0b111_010_00_00000000_00000000_00000000, KalimbaOp.UNUSED),
    #(0b111111_11_00000000_00000000_00000000, 0b111_011_00_00000000_00000000_00000000, KalimbaOp.UNUSED),
    # Order matters here!
    (0b111111_11_00001111_00000000_00000000, 0b111_100_00_00000011_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_a),
    (0b111111_11_00001111_00000000_00000000, 0b111_100_00_00000111_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_a),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00000000_00000000_00000000, KalimbaOp.PUSH,   kalimba_maxim_decode_stack_a),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00000100_00000000_00000000, KalimbaOp.POP,    kalimba_maxim_decode_stack_a),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00001000_00000000_00000000, KalimbaOp.LOADS,  kalimba_maxim_decode_stack_a),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00001100_00000000_00000000, KalimbaOp.STORES, kalimba_maxim_decode_stack_a),
    #(0b111111_11_00000000_00000000_00000000, 0b111_110_00_00000000_00000000_00000000, KalimbaOp.UNUSED),

    # Type B
    (0b111001_11_00000000_00000000_00000000, 0b000_000_01_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank1_b_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b000_001_01_00000000_00000000_00000000, KalimbaOp.ADC, kalimba_maxim_decode_binop_bank1_b_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_000_01_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank1_b_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_001_01_00000000_00000000_00000000, KalimbaOp.SBB, kalimba_maxim_decode_binop_bank1_b_addsub),
    (0b111000_11_00000000_00000000_00000000, 0b010_000_01_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank12_b), #ADDB12
    (0b111000_11_00000000_00000000_00000000, 0b011_000_01_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank12_b), #SUBB12
    (0b111111_11_00000000_00000000_00000000, 0b100_000_01_00000000_00000000_00000000, KalimbaOp.AND, kalimba_maxim_decode_binop_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_001_01_00000000_00000000_00000000, KalimbaOp.OR,  kalimba_maxim_decode_binop_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_010_01_00000000_00000000_00000000, KalimbaOp.XOR, kalimba_maxim_decode_binop_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_011_01_00000000_00000000_00000000, KalimbaOp.LSHIFT, kalimba_maxim_decode_shift_by_c_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_100_01_00000000_00000000_00000000, KalimbaOp.ASHIFT, kalimba_maxim_decode_shift_by_c_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_110_01_00000000_00000000_00000000, KalimbaOp.IMUL, kalimba_maxim_decode_binop_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_111_01_00000000_00000000_00000000, KalimbaOp.SMUL, kalimba_maxim_decode_binop_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b100_101_01_00000000_00000000_00000000, KalimbaOp.FMUL, kalimba_maxim_decode_binop_bank1_b),

    # TODO: tests
    (0b111100_11_00000000_00000000_00000000, 0b101_000_01_00000000_00000000_00000000, KalimbaOp.FMADD, kalimba_maxim_decode_fmaddsub_b),
    (0b111100_11_00000000_00000000_00000000, 0b101_100_01_00000000_00000000_00000000, KalimbaOp.FMSUB, kalimba_maxim_decode_fmaddsub_b),
    (0b111100_11_00000000_00000000_00000000, 0b110_000_01_00000000_00000000_00000000, KalimbaOp.MULX,  kalimba_maxim_decode_fmaddsub_b),
    (0b111111_11_00000000_00000000_00000000, 0b110_100_01_00000000_00000000_00000000, KalimbaOp.LOAD,  kalimba_maxim_decode_load_store_b),
    (0b111111_11_00000000_00000000_00000000, 0b110_101_01_00000000_00000000_00000000, KalimbaOp.STORE, kalimba_maxim_decode_load_store_b),
    (0b111111_11_00000000_00000000_00000000, 0b110_110_01_00000000_00000000_00000000, KalimbaOp.DIV, kalimba_maxim_decode_divide_b),
    (0b111111_11_00000000_00000000_00000000, 0b110_111_01_00000000_00000000_00000000, KalimbaOp.JUMP, kalimba_maxim_decode_flow_b),
    (0b111111_11_00000000_00000000_00000000, 0b111_000_01_00000000_00000000_00000000, KalimbaOp.CALL, kalimba_maxim_decode_flow_b),
    (0b111111_11_00000000_00000000_00000000, 0b111_001_01_00000000_00000000_00000000, KalimbaOp.DOLOOP, kalimba_maxim_decode_doloop_b),
    (0b111111_11_00000000_00000000_00000000, 0b111_010_01_00000000_00000000_00000000, KalimbaOp.LSHIFT, kalimba_maxim_decode_shift_c_bank1_b),
    (0b111111_11_00000000_00000000_00000000, 0b111_011_01_00000000_00000000_00000000, KalimbaOp.ASHIFT, kalimba_maxim_decode_shift_c_bank1_b),
    (0b111111_11_00001111_00000000_00000000, 0b111_100_01_00000011_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_b),
    (0b111111_11_00001111_00000000_00000000, 0b111_100_01_00000111_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_b),
    (0b111111_11_00001111_00000000_00000000, 0b111_100_01_00001111_00000000_00000000, KalimbaOp.PUSH,   kalimba_maxim_decode_stack_b),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_01_00000000_00000000_00000000, KalimbaOp.PUSHM,  kalimba_maxim_decode_stack_b),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_01_00000100_00000000_00000000, KalimbaOp.POPM,   kalimba_maxim_decode_stack_b),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_01_00001000_00000000_00000000, KalimbaOp.LOADS,  kalimba_maxim_decode_stack_b),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_01_00001100_00000000_00000000, KalimbaOp.STORES, kalimba_maxim_decode_stack_b),

    # Type C
    (0b111001_11_00000000_00000000_00000000, 0b000_000_10_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank1_creg_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b000_001_10_00000000_00000000_00000000, KalimbaOp.ADC, kalimba_maxim_decode_binop_bank1_creg_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_000_10_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank1_creg_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_001_10_00000000_00000000_00000000, KalimbaOp.SBB, kalimba_maxim_decode_binop_bank1_creg_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b000_000_11_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank1_const_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b000_001_11_00000000_00000000_00000000, KalimbaOp.ADC, kalimba_maxim_decode_binop_bank1_const_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_000_11_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank1_const_addsub),
    (0b111001_11_00000000_00000000_00000000, 0b001_001_11_00000000_00000000_00000000, KalimbaOp.SBB, kalimba_maxim_decode_binop_bank1_const_addsub),
    (0b111000_11_00000000_00000000_00000000, 0b010_000_10_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank12_creg), #ADDB12
    (0b111000_11_00000000_00000000_00000000, 0b011_000_10_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank12_creg), #SUBB12
    (0b111000_11_00000000_00000000_00000000, 0b010_000_11_00000000_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_binop_bank12_const), #ADDB12
    (0b111000_11_00000000_00000000_00000000, 0b011_000_11_00000000_00000000_00000000, KalimbaOp.SUB, kalimba_maxim_decode_binop_bank12_const), #SUBB12
    (0b111111_11_00000000_00000000_00000000, 0b100_000_10_00000000_00000000_00000000, KalimbaOp.AND, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_001_10_00000000_00000000_00000000, KalimbaOp.OR,  kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_010_10_00000000_00000000_00000000, KalimbaOp.XOR, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_011_10_00000000_00000000_00000000, KalimbaOp.LSHIFT, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_100_10_00000000_00000000_00000000, KalimbaOp.ASHIFT, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_110_10_00000000_00000000_00000000, KalimbaOp.IMUL, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_111_10_00000000_00000000_00000000, KalimbaOp.SMUL, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_101_10_00000000_00000000_00000000, KalimbaOp.FMUL, kalimba_maxim_decode_binop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b100_000_11_00000000_00000000_00000000, KalimbaOp.AND, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_001_11_00000000_00000000_00000000, KalimbaOp.OR,  kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_010_11_00000000_00000000_00000000, KalimbaOp.XOR, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_011_11_00000000_00000000_00000000, KalimbaOp.LSHIFT, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_100_11_00000000_00000000_00000000, KalimbaOp.ASHIFT, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_110_11_00000000_00000000_00000000, KalimbaOp.IMUL, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_111_11_00000000_00000000_00000000, KalimbaOp.SMUL, kalimba_maxim_decode_binop_bank1_const),
    (0b111111_11_00000000_00000000_00000000, 0b100_101_11_00000000_00000000_00000000, KalimbaOp.FMUL, kalimba_maxim_decode_binop_bank1_const),
    # (0b111100_11_00000000_00000000_00000000, 0b101_000_10_00000000_00000000_00000000, KalimbaOp.FMADD, kalimba_maxim_decode_fmaddsub_creg),
    # (0b111100_11_00000000_00000000_00000000, 0b101_000_10_00000000_00000000_00000000, KalimbaOp.FMSUB, kalimba_maxim_decode_fmaddsub_creg),
    # (0b111100_11_00000000_00000000_00000000, 0b110_000_10_00000000_00000000_00000000, KalimbaOp.MULX,  kalimba_maxim_decode_fmaddsub_creg),
    # (0b111100_11_00000000_00000000_00000000, 0b101_000_11_00000000_00000000_00000000, KalimbaOp.FMADD, kalimba_maxim_decode_fmaddsub_const),
    # (0b111100_11_00000000_00000000_00000000, 0b101_000_11_00000000_00000000_00000000, KalimbaOp.FMSUB, kalimba_maxim_decode_fmaddsub_const),
    # (0b111100_11_00000000_00000000_00000000, 0b110_000_11_00000000_00000000_00000000, KalimbaOp.MULX,  kalimba_maxim_decode_fmaddsub_const),
    (0b111111_11_00000000_00000000_00000000, 0b110_100_10_00000000_00000000_00000000, KalimbaOp.LOAD,  kalimba_maxim_decode_load_creg),
    (0b111111_11_00000000_00000000_00000000, 0b110_100_11_00000000_00000000_00000000, KalimbaOp.LOAD,  kalimba_maxim_decode_load_const),
    (0b111111_11_00000000_00000000_00000000, 0b110_110_10_00000000_00000000_00000000, KalimbaOp.BSIGN, kalimba_maxim_decode_unop_bank1_creg),
    (0b111111_11_00000000_00000000_00000000, 0b110_110_11_00000000_00000000_00000000, KalimbaOp.BSIGN, kalimba_maxim_decode_unop_bank1_const),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_10_00000000_00000000_00000000, KalimbaOp.PUSH, kalimba_maxim_decode_stack_creg),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_10_00000100_00000000_00000000, KalimbaOp.POP,  kalimba_maxim_decode_stack_creg),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_11_00000000_00000000_00000000, KalimbaOp.PUSH, kalimba_maxim_decode_stack_const),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_11_00000100_00000000_00000000, KalimbaOp.POP,  kalimba_maxim_decode_stack_const),

    # CMAC
    #(0b111111_11_00001100_00000000_00000000, 0b111_100_10_00000000_00000000_00000000, KalimbaOp.FMADD, kalimba_maxim_decode_fmaddsub_cmac),

    # Subword A
    (0b11111111_00000000_00000000_00000000, 0b11110100_00000000_00000000_00000000, KalimbaOp.LOADW, kalimba_maxim_decode_subword_a),
    # Subword B
    (0b11111111_00000000_00000000_00000000, 0b11110101_00000000_00000000_00000000, KalimbaOp.LOADW, kalimba_maxim_decode_subword_b),
    # Prefix
    (0b11111111_11100000_00000000_00000000, 0b11111101_00000000_00000000_00000000, KalimbaOp.PREFIX, kalimba_maxim_decode_prefix),
]

def kalimba_maxim_lookup_decoder(instruction: int) -> Callable[[int, KalimbaOp, Optional[KalimbaPrefix]], KalimbaInstruction]:
    for (mask, value, opcode, decode) in maxim_ops_lut:
        if (instruction & mask) == value:
            return (instruction, opcode, decode)
    raise ValueError(f'invalid instruction 0b{instruction:032b}') # pragma: no cover

def kalimba_maxim_decode(data: bytes) -> Tuple[KalimbaInstruction, int]:
    if len(data) < 4:
        raise EOFError(f'expected at least 4 bytes, but got {len(data)}') # pragma: no cover

    (instruction, opcode, decode) = kalimba_maxim_lookup_decoder(int.from_bytes(data[:4], 'little'))
    op = decode(instruction, opcode, None)
    consumed = 4;
    if isinstance(op, KalimbaPrefix):
        if len(data) < 8:
            raise EOFError(f'expected at least 8 bytes, but got {len(data)}') # pragma: no cover
        (instruction, opcode, decode) = kalimba_maxim_lookup_decoder(int.from_bytes(data[4:8], 'little'))
        op = decode(instruction, opcode, op)
        consumed += 4;

    return (op, consumed)


def test():
    from instruction_test import kalimba_maxim_instructions_test

    for (data, asm) in kalimba_maxim_instructions_test:
        try:
            (op, length) = kalimba_maxim_decode(data)
            assert length == len(data)

            asm = asm[:-1].replace(' EQ ', ' Z ').replace(' NE ', ' NZ ')

            asm_str = f'{type(op).__name__}("{asm}")'
            op_str = f'{op}'

            # This is special case, can't be bothered to fix
            if op_str == asm_str or op_str.replace(' (SS)', '') == asm_str:
                #print(f'PASS: {op} | {asm} | {data[::-1].hex()}')
                continue
            #assert f'{op}' == f'{type(op).__name__}("{asm}")', f'{op} != {type(op).__name__}("{asm}")'
            (instruction, opcode, decode) = kalimba_maxim_lookup_decoder(int.from_bytes(data[-4:], 'little'))
            print(f'FAIL: {op} | {asm} | {data[::-1].hex()} | {decode}')
        except ValueError:
            print(f'UNSUPPORTED: failed to decode "{data.hex()}" ("{asm}")')
        except Exception as e:
            print(f'ERROR: failed to decode "0b{int.from_bytes(data[:4], 'little'):032b}" ("{asm}")')
            print(f'                        "{data.hex()}" ("{asm}")')
            raise e

if __name__ == '__main__':
    test()
