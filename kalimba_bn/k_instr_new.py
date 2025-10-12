from enum import IntEnum, auto
from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union
from dataclasses import dataclass
from functools import partial

class KalimbaRegBase(IntEnum):
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
class KalimbaMemAccess:
    '''
    Reads/writes one word
    '''
    sub: KalimbaSubWordMem
    reg: KalimbaReg
    offset: int
    def __str__(self):
        t = self.sub.name[2:]
        if self.offset == 0:
            return f'{t}[{self.reg}]'
        elif self.offset >= 0:
            return f'{t}[{self.reg} + {self.offset}]'
        else:
            return f'{t}[{self.reg} - {-self.offset}]'

@dataclass(unsafe_hash=True)
class KalimbaIndexedMemAccess:
    '''
    Reads/writes one word, post increment index by modify
    '''
    write: bool
    reg: KalimbaReg                  # Any register
    idx: KalimbaBank2Reg             # Ix
    mod: Union[KalimbaBank2Reg, int] # Mx or constant
    def __str__(self):
        m = f'M[{self.idx},{self.mod}]'
        if self.write:
            return f'{m} = {self.reg}'
        else:
            return f'{self.reg} = {m}'

type KalimbaOperand = Union[KalimbaReg, KalimbaMemAccess, int]

reg_bank_lut = {
    1: KalimbaBank1Reg,
    2: KalimbaBank2Reg,
    3: KalimbaBank3Reg,
}

def get_reg(bank, index):
    return reg_bank_lut[bank](index)

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
    FMAAS  = auto() # Multiply-accumulate + Add/Sub
    SIGN   = auto() # Sign detect / Block sign detect
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
    FMAASB = auto() # Multiply-accumulate + Add/Sub
    LSHC   = auto() # Logical shift of a constant
    ASHC   = auto() # Arithmetic shift of a constant
    FMSAS  = auto() # Multiply-subtract + Add/Sub
    MULASB = auto() # Multiply + Add/Sub
    PUSH   = auto() # Push to stack
    POP    = auto() # Pop from stack
    LOADS  = auto() # Load to stack with offset
    STORES = auto() # Store to stack with offset
    LOADW  = auto() # Subword load
    STOREW = auto() # Subword store
    PREFIX = auto() # 32-bit constant prefix
    UNUSED = auto() # Reserved

@dataclass(unsafe_hash=True)
class KalimbaUnOp:
    '''
    C = OP A
    '''
    op: KalimbaOp
    c: KalimbaOperand
    a: KalimbaOperand
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]

    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        if self.cond == KalimbaCond.Always:
            return f'KalimbaUnOp("{self.c} = {self.op.name} {self.a}{m}")'
        else:
            return f'KalimbaUnOp("if {self.cond.name} {self.c} = {self.op.name} {self.a}{m}")'

binop_symbols = {
    KalimbaOp.ADD:  ('+', ''),
    KalimbaOp.ADC:  ('+', ' + Carry'),
    KalimbaOp.SUB:  ('-', ''),
    KalimbaOp.SBB:  ('-', ' - Borrow'),
    KalimbaOp.IMUL: ('*', ' (int)'),
    KalimbaOp.SMUL: ('*', ' (int) (sat)'),
    KalimbaOp.FMUL: ('*', ' (frac)'),
}

@dataclass(unsafe_hash=True)
class KalimbaBinOp:
    '''
    C = A OP B
    '''
    op: KalimbaOp
    c: KalimbaOperand
    a: KalimbaOperand
    b: KalimbaOperand
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]

    def __str__(self):
        op = self.op.name
        extra = ''
        if self.op in binop_symbols:
            (op, extra) = binop_symbols[self.op]

        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '

        return f'KalimbaBinOp("{cond}{self.c} = {self.a} {op} {self.b}{extra}{m}")'

def get_mask(length):
    return (1 << length) - 1

def get_bits(instruction, offset, length):
    return (instruction >> offset) & get_mask(length)

def kalimba_maxim_decode_no_regc_a(instruction: int, banka = KalimbaBank1Reg, bankb = KalimbaBank1Reg):
    cond   = KalimbaCond(get_bits(instruction, 0, 4))
    regb   = bankb(get_bits(instruction, 4, 4))
    mag1   = KalimbaBank2Reg(KalimbaBank2Reg.M0.value + get_bits(instruction, 8, 2))
    iag1   = KalimbaBank2Reg(KalimbaBank2Reg.I0.value + get_bits(instruction, 10, 2))
    regag1 = KalimbaBank1Reg(get_bits(instruction, 12, 3))
    ag1w   = bool(get_bits(instruction, 15, 1))
    rega   = banka(get_bits(instruction, 16, 4))

    mem = KalimbaIndexedMemAccess(ag1w, regag1, iag1, mag1) if regag1 != KalimbaBank1Reg.Null else None

    return (cond, regb, mem, rega)

def kalimba_maxim_decode_a(instruction: int, banka = KalimbaBank1Reg, bankb = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    (cond, regb, mem, rega) = kalimba_maxim_decode_no_regc_a(instruction, banka, bankb)
    regc = bankc(get_bits(instruction, 20, 4))

    return (cond, regb, mem, rega, regc)

def kalimba_maxim_decode_unop_bank1_a(instruction: int, op: KalimbaOp):
    (cond, regb, mem, rega, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaUnOp(op, regc, rega, cond, mem)

def kalimba_maxim_decode_binop_bank1_a(instruction, op):
    (cond, regb, mem, rega, regc) = kalimba_maxim_decode_a(instruction)

    # Special case: rFlags is invalid here, this is actually
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT] and regc is KalimbaBank1Reg.rFlags:
        regc = KalimbaBank1Reg.rMAC

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

class KalimbaAddressingMode(IntEnum):
    RRR = 0b00 # reg = reg OP reg/imm
    RRM = 0b01 # reg = reg OP mem[reg/imm]
    RMR = 0b10 # reg = mem[reg] OP reg/imm
    MRR = 0b11 # mem[r/imm] = reg OP reg

R = lambda r: r
ML = lambda r: KalimbaMemAccess(KalimbaSubWordMem.L_M, r, 0)
MS = lambda r: KalimbaMemAccess(KalimbaSubWordMem.S_M, r, 0)

addressing_mode_lut = {
    0b00: ( R,  R,  R), # reg = reg OP reg/imm
    0b01: ( R,  R, ML), # reg = reg OP mem[reg/imm]
    0b10: ( R, ML,  R), # reg = mem[reg] OP reg/imm
    0b11: (MS,  R,  R), # mem[r/imm] = reg OP reg
}

def kalimba_maxim_decode_binop_bank1_a_addsub(instruction, op):
    (cond, regb, mem,  rega, regc) = kalimba_maxim_decode_a(instruction)
    (fc, fa, fb) = addressing_mode_lut[get_bits(instruction, 27, 2)]
    return KalimbaBinOp(op, fc(regc), fa(rega), fb(regb), cond, mem)

def kalimba_maxim_decode_binop_bank1_a_const(const, instruction, op):
    (cond, regb, mem, rega, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaBinOp(op, regc, rega, const, cond, mem)

kalimba_maxim_decode_binop_bank1_a_const1 = partial(kalimba_maxim_decode_binop_bank1_a_const, 1)
kalimba_maxim_decode_binop_bank1_a_const2 = partial(kalimba_maxim_decode_binop_bank1_a_const, 2)
kalimba_maxim_decode_binop_bank1_a_const4 = partial(kalimba_maxim_decode_binop_bank1_a_const, 4)

bank_select_lut = {
    0b000: (KalimbaBank1Reg, KalimbaBank1Reg, KalimbaBank1Reg), # B1 = B1 +- B1 | B1 +-  K
    0b001: (KalimbaBank1Reg, KalimbaBank1Reg, KalimbaBank2Reg), # B1 = B1 +- B2 |  K  - B1
    0b010: (KalimbaBank1Reg, KalimbaBank2Reg, KalimbaBank1Reg), # B1 = B2 +- B1 | B2 +-  K
    0b011: (KalimbaBank1Reg, KalimbaBank2Reg, KalimbaBank2Reg), # B1 = B2 +- B2 |  K  - B2
    0b100: (KalimbaBank2Reg, KalimbaBank1Reg, KalimbaBank1Reg), # B2 = B1 +- B1 | B1 +-  K
    0b101: (KalimbaBank2Reg, KalimbaBank1Reg, KalimbaBank2Reg), # B2 = B1 +- B2 |  K  - B1
    0b110: (KalimbaBank2Reg, KalimbaBank2Reg, KalimbaBank1Reg), # B2 = B2 +- B1 | B2 +- K
    0b111: (KalimbaBank2Reg, KalimbaBank2Reg, KalimbaBank2Reg), # B2 = B2 +- B2 |  K  - B2
}

assert KalimbaBank1Reg.rFlags == KalimbaBank2Reg.L4
assert KalimbaBank1Reg.rFlags is not KalimbaBank2Reg.L4

def kalimba_maxim_decode_binop_bank12_a(instruction, op):
    (bankc, banka, bankb) = bank_select_lut[get_bits(instruction, 26, 3)]
    (cond, regb, mem, rega, regc) = kalimba_maxim_decode_a(instruction, banka, bankb, bankc)

    # Special case: rFlags is invalid here, this is actually FP
    if rega is KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    if regb is KalimbaBank1Reg.rFlags:
        regb = KalimbaBank3Reg.FP

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

class KalimbaSignSelect(IntEnum):
    UU = 0b00
    US = 0b01
    SU = 0b10
    SS = 0b11

@dataclass(unsafe_hash=True)
class KalimbaExtraAddSub:
    op: Union[KalimbaOp.ADD, KalimbaOp.SUB]
    a: KalimbaOperand
    b: KalimbaOperand

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
    C = C OP A * B [r = D OP E]
    '''
    op: KalimbaOp
    c: Union[KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB]
    a: KalimbaBank1Reg
    b: KalimbaBank1Reg
    cond: KalimbaCond
    sign: Optional[KalimbaSignSelect]
    addsub: Optional[KalimbaBinOp]
    mem: Optional[KalimbaIndexedMemAccess]

    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        addsub = f', {self.addsub}' if self.addsub else ''
        sign = f' ({self.sign.name})' if self.sign else ''

        if self.op in fused_mul_symbols:
            op = fused_mul_symbols[self.op]
            return f'KalimbaBinOp("{cond}{self.c} = {self.c} {op} {self.a} * {self.b}{sign}{addsub}{m}")'
        else:
            return f'KalimbaBinOp("{cond}{self.c} = {self.a} * {self.b}{sign}{addsub}{m}")'

def kalimba_maxim_decode_fmaddsub_a(instruction, op):
    (cond, regb, mem, rega) = kalimba_maxim_decode_no_regc_a(instruction)

    addsub = None
    sign = None

    if get_bits(instruction, 23, 1) == 0:
        # RegC' = RegC' OP RegA * RegB
        regc = KalimbaBank1Reg(get_bits(instruction, 20, 3))
        # Special case: Null is actually rMACB
        if regc is KalimbaBank1Reg.Null:
            regc = KalimbaBank1Reg.rMACB
        sign = KalimbaSignSelect(get_bits(instruction, 26, 2))
    else:
        # RegC'' = RegC'' OP RegA * RegB, r0 = RegD OP RegE
        regc = [KalimbaBank1Reg.rMACB, KalimbaBank1Reg.rMAC, KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 20, 2)]
        regd = [KalimbaBank1Reg.r1, KalimbaBank1Reg.r2][get_bits(instruction, 25, 1)]
        rege = [KalimbaBank1Reg.rMAC, KalimbaBank1Reg.rMACB][get_bits(instruction, 22, 1)]
        addsub_op = [KalimbaOp.ADD, KalimbaOp.SUB][get_bits(instruction, 24, 1)]
        addsub = KalimbaExtraAddSub(addsub_op, regd, rege)

    return KalimbaFusedMultiplyAddSub(op, regc, rega, regb, cond, sign, addsub, mem)

# mask, value, operation, decode
maxim_ops_lut = [
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
    (0b111100_11_00000000_00000000_00000000, 0b110_000_00_00000000_00000000_00000000, KalimbaOp.MULX),
    (0b111111_11_00000000_00000000_00000000, 0b110_100_00_00000000_00000000_00000000, KalimbaOp.LOAD),
    (0b111111_11_00000000_00000000_00000000, 0b110_101_00_00000000_00000000_00000000, KalimbaOp.STORE),
    (0b111111_11_00000000_00000000_00000000, 0b110_110_00_00000000_00000000_00000000, KalimbaOp.SIGN),
    (0b111111_11_00000000_00000000_00000000, 0b110_111_00_00000000_00000000_00000000, KalimbaOp.JUMP), # Special cases: RTS/RTI
    (0b111111_11_00000000_00000000_00000000, 0b111_000_00_00000000_00000000_00000000, KalimbaOp.CALL),
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

    # TODO: push/pop
    #(0b111111_11_00000000_00000000_00000000, 0b111_100_00_00000000_00000000_00000000, KalimbaOp.),

    # TODO: subword
    #(0b111111_11_00000000_00000000_00000000, 0b111_101_00_00000000_00000000_00000000, KalimbaOp.),

    #(0b111111_11_00000000_00000000_00000000, 0b111_110_00_00000000_00000000_00000000, KalimbaOp.UNUSED),

    # Type B
    #(0b111111_11_00000000_00000000_00000000, 0b _ _01_00000000_00000000_00000000, KalimbaOp.),
    # Type C
    #(),
    # Subword A
    (0b11111111_00000000_00000000_00000000, 0b11110100_00000000_00000000_00000000, KalimbaOp.LOADW),
    # Subword B
    (0b11111111_00000000_00000000_00000000, 0b11110101_00000000_00000000_00000000, KalimbaOp.LOADW),
    # Prefix
    (0b11111111_11100000_00000000_00000000, 0b11111101_00000000_00000000_00000000, KalimbaOp.PREFIX),
]

type KalimbaInstruction = Union[KalimbaUnOp, KalimbaBinOp, KalimbaFusedMultiplyAddSub]

def kalimba_maxim_lookup_op(instruction: int) -> KalimbaInstruction:
    for (mask, value, opcode, *other) in maxim_ops_lut:
        if (instruction & mask) == value:
            return opcode.name if not other else other[0](instruction, opcode)
    raise ValueError(f'invalid instruction 0b{instruction:032b}')

if __name__ == '__main__':
    print(kalimba_maxim_lookup_op(0x5812002f))# 2f 00 12 58 | I1 = I2 + r0;
    print(kalimba_maxim_lookup_op(0x0034005f))# 5f 00 34 00 | r1 = r2 + r3;
    print(kalimba_maxim_lookup_op(0x0434005f))# 5f 00 34 04 | r1 = r2 + r3 + Carry;
    print(kalimba_maxim_lookup_op(0x0035004f))# 4f 00 35 00 | r1 = r3 + r2;
    print(kalimba_maxim_lookup_op(0x2034005f))# 5f 00 34 20 | r1 = r2 - r3;
    print(kalimba_maxim_lookup_op(0x2035004f))# 4f 00 35 20 | r1 = r3 - r2;
    print(kalimba_maxim_lookup_op(0x2435004f))# 4f 00 35 24 | r1 = r3 - r2 - Borrow;

    print(kalimba_maxim_lookup_op(0x1834005f))# 5f 00 34 18 | M[r1] = r2 + r3;
    print(kalimba_maxim_lookup_op(0x18340056))# 56 00 34 18 | if V M[r1] = r2 + r3;

    print(kalimba_maxim_lookup_op(0x18342456))# 56 24 34 18 | if V M[r1] = r2 + r3, r0 = M[I1,M0];

    print(kalimba_maxim_lookup_op(0xe434454f))# 4f 45 34 e4 | r1 = ABS r2, r2 = M[I1,M1];
    print(kalimba_maxim_lookup_op(0xe434ba4f))# 4f ba 34 e4 | r1 = ABS r2, M[I2,M2] = r1;

    print(kalimba_maxim_lookup_op(0x8834005f))# 5f 00 34 88 | r1 = r2 XOR r3;

    print(kalimba_maxim_lookup_op(0x8cea00b1))# b1 00 ea 8c | if Z rMAC = r8 LSHIFT r9;
    print(kalimba_maxim_lookup_op(0x90e70090))# 90 00 e7 90 | if EQ rMAC = r5 ASHIFT r7;

    print(kalimba_maxim_lookup_op(0x88170090))# 90 00 17 88 | if EQ rMAC = r5 XOR r7;
    print(kalimba_maxim_lookup_op(0x00170090))# 90 00 17 00 | if EQ rMAC = r5 + r7;
    print(kalimba_maxim_lookup_op(0x88e70090))# 90 00 e7 88 | if EQ rFlags = r5 XOR r7;

    print(kalimba_maxim_lookup_op(0x9434005f))# 5f 00 34 94 | r1 = r2 * r3 (frac);
    print(kalimba_maxim_lookup_op(0x9834005f))# 5f 00 34 98 | r1 = r2 * r3 (int);
    print(kalimba_maxim_lookup_op(0x9c34005f))# 5f 00 34 9c | r1 = r2 * r3 (int) (sat);
    print(kalimba_maxim_lookup_op(0x9cf4005f))# 5f 00 f4 9c | rMACB = r2 * r3 (int) (sat);

    print(kalimba_maxim_lookup_op(0xac23504f))# 4f 50 23 ac | r0 = r0 + r1 * r2 (SS), r3 = M[I0,M0];
    print(kalimba_maxim_lookup_op(0xa4c3504f))# 4f 50 c3 a4 | rMACB = rMACB + r1 * r2, r0 = r1 - rMACB, r3 = M[I0,M0];
    print(kalimba_maxim_lookup_op(0xacc5906f))# 6f 90 c5 ac | rMACB = rMACB + r3 * r4, r0 = r2 - rMACB, M[I0,M0] = rMAC;

    print(kalimba_maxim_lookup_op(0xa483504f))# 4f 50 83 a4 | rMACB = rMACB + r1 * r2, r0 = r1 - rMAC, r3 = M[I0,M0];
    print(kalimba_maxim_lookup_op(0xac85906f))# 6f 90 85 ac | rMACB = rMACB + r3 * r4, r0 = r2 - rMAC, M[I0,M0] = rMAC;
    print(kalimba_maxim_lookup_op(0xb0d3504f))# 4f 50 d3 b0 | rMAC = rMAC - r1 * r2, r0 = r1 + rMACB, r3 = M[I0,M0];
    print(kalimba_maxim_lookup_op(0xb8d5906f))# 6f 90 d5 b8 | rMAC = rMAC - r3 * r4, r0 = r2 + rMACB, M[I0,M0] = rMAC;
    print(kalimba_maxim_lookup_op(0xa803504e))# 4e 50 03 a8 | if USERDEF rMACB = rMACB + r1 * r2 (SU), r3 = M[I0,M0];

    print(kalimba_maxim_lookup_op(0x581200ef))#ef 00 12 58 | I1 = I2 + FP;
    print(kalimba_maxim_lookup_op(0x541e002f))#2f 00 1e 54 | I1 = FP + I2;
    print(kalimba_maxim_lookup_op(0x4ce1002f))#2f 00 e1 4c | rFlags = I1 + I2;

    print(kalimba_maxim_lookup_op(0xdc0d000f))# 0f 00 0d dc | rts; // special case of jump
    print(kalimba_maxim_lookup_op(0xe434002f))# 2f 00 34 e4 | r1 = r2 + 1;
    print(kalimba_maxim_lookup_op(0xe434003f))# 3f 00 34 e4 | r1 = r2 - 1;
    print(kalimba_maxim_lookup_op(0xe434004f))# 4f 00 34 e4 | r1 = ABS r2;
    print(kalimba_maxim_lookup_op(0xe434005f))# 5f 00 34 e4 | r1 = MIN r2;
    print(kalimba_maxim_lookup_op(0xe434006f))# 6f 00 34 e4 | r1 = MAX r2;
    print(kalimba_maxim_lookup_op(0xe434007f))# 7f 00 34 e4 | r1 = TWOBITCOUNT r2;
    print(kalimba_maxim_lookup_op(0xe434008f))# 8f 00 34 e4 | r1 = MOD24 r2;
    print(kalimba_maxim_lookup_op(0xe43400af))# af 00 34 e4 | r1 = r2 + 2;
    print(kalimba_maxim_lookup_op(0xe43400bf))# bf 00 34 e4 | r1 = r2 + 4;
    print(kalimba_maxim_lookup_op(0xe43400cf))# cf 00 34 e4 | r1 = r2 - 2;
    print(kalimba_maxim_lookup_op(0xe43400df))# df 00 34 e4 | r1 = r2 - 4;
    print(kalimba_maxim_lookup_op(0xe43400ef))# ef 00 34 e4 | r1 = SE8 r2;
    print(kalimba_maxim_lookup_op(0xe43400ff))# ff 00 34 e4 | r1 = SE16 r2;
    #print(kalimba_maxim_lookup_op(0x03000000))# Type C
    #print(kalimba_maxim_lookup_op(0x23000000))
    #print(kalimba_maxim_lookup_op(0x01b00004))# Type B
    #print(kalimba_maxim_lookup_op(0xe434000f))# 0f 00 34 e4 | kalcode(e434000f);
    #print(kalimba_maxim_lookup_op(0xe434001f))# 1f 00 34 e4 | kalcode(e434001f);
