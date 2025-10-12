from enum import IntEnum, auto
from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union, Literal
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
    POP    = auto() # Pop from stack
    LOADS  = auto() # Load from stack with offset
    STORES = auto() # Store to stack with offset
    LOADW  = auto() # Subword load
    STOREW = auto() # Subword store
    PREFIX = auto() # 32-bit constant prefix
    UNUSED = auto() # Reserved

unop_symbols = {
    KalimbaOp.SIGN:  'SIGNDET',
    KalimbaOp.BSIGN: 'BLKSIGNDET',
}

@dataclass(unsafe_hash=True)
class KalimbaUnOp:
    '''
    C = OP A
    '''
    op: KalimbaOp
    c: KalimbaReg
    a: KalimbaReg
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]

    def __str__(self):
        op = self.op.name

        if self.op == KalimbaOp.DIV:
            assert self.mem == None and self.cond == KalimbaCond.Always
            return f'Div = {self.c} / {self.a}'

        if self.op in unop_symbols:
            op = unop_symbols[self.op]

        m = f', {self.mem}' if self.mem else ''
        if self.cond == KalimbaCond.Always:
            return f'KalimbaUnOp("{self.c} = {op} {self.a}{m}")'
        else:
            return f'KalimbaUnOp("if {self.cond.name} {self.c} = {op} {self.a}{m}")'

binop_symbols = {
    KalimbaOp.ADD:  ('+', ''),
    KalimbaOp.ADC:  ('+', ' + Carry'),
    KalimbaOp.SUB:  ('-', ''),
    KalimbaOp.SBB:  ('-', ' - Borrow'),
    KalimbaOp.IMUL: ('*', ' (int)'),
    KalimbaOp.SMUL: ('*', ' (int) (sat)'),
    KalimbaOp.FMUL: ('*', ' (frac)'),
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
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]
    shift: Optional[KalimbaShiftType] = None

    def __str__(self):
        op = self.op.name
        extra = ''
        if self.op in binop_symbols:
            (op, extra) = binop_symbols[self.op]
        elif self.op in [KalimbaOp.ASHIFT, KalimbaOp.LSHIFT] and self.shift:
            extra = f' {shift_type_lut[self.shift]}'

        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '

        return f'KalimbaBinOp("{cond}{self.c} = {self.a} {op} {self.b}{extra}{m}")'

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

def kalimba_maxim_decode_unop_bank1_a(instruction: int, op: KalimbaOp):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaUnOp(op, regc, rega, cond, mem)

def kalimba_maxim_decode_binop_bank1_a(instruction, op):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)

    shift = None

    # Special case: rFlags is invalid here, so this is actually rMAC with 32-bit width
    if op in [KalimbaOp.LSHIFT, KalimbaOp.ASHIFT]:
        if regc is KalimbaBank1Reg.rFlags:
            regc = KalimbaBank1Reg.rMAC
            shift = KalimbaShiftType.ST_32
        elif regc is KalimbaBank1Reg.rMAC or regc is KalimbaBank1Reg.rMACB:
            shift = KalimbaShiftType.ST_72

    return KalimbaBinOp(op, regc, rega, regb, cond, mem, shift)

def kalimba_maxim_decode_b(instruction: int, banka = KalimbaBank1Reg, bankc = KalimbaBank1Reg):
    k16 = get_bits_signed(instruction, 0, 16)
    rega = banka(get_bits(instruction, 16, 4))
    regc = bankc(get_bits(instruction, 20, 4))

    return (k16, rega, regc)

def kalimba_maxim_decode_binop_bank1_b(instruction, op):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    return KalimbaBinOp(op, regc, rega, k16, KalimbaCond.Always, None)


def kalimba_maxim_decode_shift_common_bank1_b(instruction):
    (_, rega, regc) = kalimba_maxim_decode_b(instruction)
    amount = get_bits_signed(instruction, 0, 8)
    dest = get_bits(instruction, 8, 3)

    shift = None

    # TODO: return as kalcode, i.e. kalcode(912b0207);
    assert regc is KalimbaBank1Reg.rMAC or regc is KalimbaBank1Reg.rMACB or dest == 0b000

    if dest == 0b001:
        shift = KalimbaShiftType.ST_LO
    elif dest == 0b000 and regc is KalimbaBank1Reg.rFlags:
        regc = KalimbaBank1Reg.rMAC
        shift = KalimbaShiftType.ST_MI
    elif dest == 0b000 and (regc is KalimbaBank1Reg.rMAC or regc is KalimbaBank1Reg.rMACB):
        shift = KalimbaShiftType.ST_72
    elif dest == 0b010:
        shift = KalimbaShiftType.ST_HI
    elif dest == 0b101:
        if regc is KalimbaBank1Reg.rMAC:
            regc = KalimbaBank3Reg.rMAC0
        elif regc is KalimbaBank1Reg.rMACB:
            regc = KalimbaBank3Reg.rMACB0
        shift = KalimbaShiftType.ST_32
    elif dest == 0b100:
        if regc is KalimbaBank1Reg.rMAC:
            #regc = KalimbaBank3Reg.rMAC12 // TODO: differentiate between rMAC(B)12 and rMAC(B)1 in __str__
            regc = KalimbaBank3Reg.rMAC1
        elif regc is KalimbaBank1Reg.rMACB:
            #regc = KalimbaBank3Reg.rMACB12
            regc = KalimbaBank3Reg.rMACB1
        shift = KalimbaShiftType.ST_32
    elif dest == 0b110:
        if regc is KalimbaBank1Reg.rMAC:
            regc = KalimbaBank3Reg.rMAC2
        elif regc is KalimbaBank1Reg.rMACB:
            regc = KalimbaBank3Reg.rMACB2
        shift = KalimbaShiftType.ST_32

    return (regc, rega, amount, shift)

def kalimba_maxim_decode_shift_by_c_bank1_b(instruction, op):
    (regc, rega, amount, shift) =  kalimba_maxim_decode_shift_common_bank1_b(instruction)
    return KalimbaBinOp(op, regc, rega, amount, KalimbaCond.Always, None, shift)

def kalimba_maxim_decode_shift_c_bank1_b(instruction, op):
    (regc, rega, amount, shift) =  kalimba_maxim_decode_shift_common_bank1_b(instruction)
    return KalimbaBinOp(op, regc, amount, rega, KalimbaCond.Always, None, shift)

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

def kalimba_maxim_decode_binop_bank1_a_addsub(instruction, op):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    (fc, fa, fb) = addressing_mode_lut[get_bits(instruction, 27, 2)]
    return KalimbaBinOp(op, fc(regc), fa(rega), fb(regb), cond, mem)

def kalimba_maxim_decode_binop_bank1_b_addsub(instruction, op):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)
    addr_mode = get_bits(instruction, 27, 2)
    (fc, fa, fb) = addressing_mode_lut[addr_mode]
    if addr_mode == 0b11:
        return KalimbaBinOp(op, fc(k16), fa(regc), fb(rega), KalimbaCond.Always, None)
    else:
        return KalimbaBinOp(op, fc(regc), fa(rega), fb(k16), KalimbaCond.Always, None)

def kalimba_maxim_decode_binop_bank1_a_const(const, instruction, op):
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

assert KalimbaBank1Reg.rFlags == KalimbaBank2Reg.L4
assert KalimbaBank1Reg.rFlags is not KalimbaBank2Reg.L4

def kalimba_maxim_decode_binop_bank12_a(instruction, op):
    (bankc, banka, bankb) = bank_select_a_lut[get_bits(instruction, 26, 3)]
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction, banka, bankb, bankc)

    # Special case: rFlags is invalid here, this is actually FP
    if rega is KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    if regb is KalimbaBank1Reg.rFlags:
        regb = KalimbaBank3Reg.FP

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

bank_select_b_lut = {
    0b00: (KalimbaBank1Reg, KalimbaBank1Reg),
    0b01: (KalimbaBank1Reg, KalimbaBank2Reg),
    0b10: (KalimbaBank2Reg, KalimbaBank1Reg),
    0b11: (KalimbaBank2Reg, KalimbaBank2Reg),
}
def kalimba_maxim_decode_binop_bank12_b(instruction, op):
    (bankc, banka) = bank_select_b_lut[get_bits(instruction, 27, 2)]
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction, banka, bankc)

    # Special case: rFlags is invalid here, this is actually FP
    if rega is KalimbaBank1Reg.rFlags:
        rega = KalimbaBank3Reg.FP

    if get_bits(instruction, 26, 1) == 0:
        return KalimbaBinOp(op, regc, rega, k16, KalimbaCond.Always, None)
    else:
        return KalimbaBinOp(op, regc, k16, rega, KalimbaCond.Always, None)

def kalimba_maxim_decode_divide_b(instruction, op):
    (_, rega, regc) = kalimba_maxim_decode_b(instruction)
    div_op = get_bits(instruction, 0, 2)

    # TODO: kalcode
    assert div != 0b11

    if div == 0b00:
        return KalimbaBinOp(op, KalimbaBank3.DivResult, regc, rega, KalimbaCond.Always, None)
    elif div == 0b01:
        return KalimbaUnOp(op, regc, KalimbaBank3.DivResult, KalimbaCond.Always, None)
    elif div == 0b10:
        return KalimbaUnOp(op, regc, KalimbaBank3.DivResult, KalimbaCond.Always, None)


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
    c: Union[Literal[KalimbaBank1Reg.rMAC], Literal[KalimbaBank1Reg.rMACB]]
    a: KalimbaBank1Reg
    b: KalimbaBank1Reg
    cond: KalimbaCond
    sign: Optional[KalimbaSignSelect]
    addsub: Optional[KalimbaExtraAddSub]
    mem: Optional[KalimbaIndexedMemAccess]

    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        addsub = f', {self.addsub}' if self.addsub else ''
        sign = f' ({self.sign.name})' if self.sign else ''

        if self.op in fused_mul_symbols:
            op = fused_mul_symbols[self.op]
            return f'KalimbaFusedMultiplyAddSub("{cond}{self.c} = {self.c} {op} {self.a} * {self.b}{sign}{addsub}{m}")'
        else:
            return f'KalimbaFusedMultiplyAddSub("{cond}{self.c} = {self.a} * {self.b}{sign}{addsub}{m}")'

def kalimba_maxim_decode_fmaddsub_a(instruction, op):
    (cond, mem, rega, regb, _) = kalimba_maxim_decode_a(instruction)

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

    return KalimbaFusedMultiplyAddSub(op, regc, rega, k16, cond, sign, addsub, mem)

def kalimba_maxim_decode_fmaddsub_b(instruction, op):
    (k16, rega, regc) = kalimba_maxim_decode_b(instruction)

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

@dataclass(unsafe_hash=True)
class KalimbaOffsetMemAccess:
    '''
    Reads/writes one word with offset
    '''
    op: KalimbaOp
    c: KalimbaBank1Reg
    a: KalimbaBank1Reg
    b: KalimbaBank1Reg
    cond: KalimbaCond
    mem: Optional[KalimbaIndexedMemAccess]
    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '

        if self.op in [KalimbaOp.LOAD, KalimbaOp.LOADS]:
            return f'KalimbaOffsetMemAccess("{cond}{self.c} = M[{self.a} + {self.b}]{m}")'
        elif self.op in [KalimbaOp.STORE, KalimbaOp.STORES]:
            return f'KalimbaOffsetMemAccess("{cond}M[{self.a} + {self.b}] = {self.c}{m}")'
        else:
            raise ValueError(f'invalid mem op: {self.op}')

def kalimba_maxim_decode_load_store_a(instruction, op):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)
    return KalimbaOffsetMemAccess(op, regc, rega, regb, cond, mem)

def kalimba_maxim_decode_load_store_b(instruction, op):
    (k16, regc, regc) = kalimba_maxim_decode_b(instruction)
    return KalimbaOffsetMemAccess(op, regc, rega, k16, KalimbaCond.Always, None)

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

        return f'KalimbaControlFlow("{cond}{n}{t}{m}")'

def kalimba_maxim_decode_flow_a(instruction, op):
    (cond, mem, rega, regb, _) = kalimba_maxim_decode_a(instruction)
    if rega is KalimbaBank1Reg.rLink:
        op = KalimbaOp.RTS
    if rega is KalimbaBank1Reg.rFlags:
        op = KalimbaOp.RTI

    return KalimbaControlFlow(op, rega, cond, mem)

def kalimba_maxim_decode_flow_b(instruction, op):
    (k16, _, regc) = kalimba_maxim_decode_b(instruction)
    cond = KalimbaCond(regc.value)

    return KalimbaControlFlow(op, k16, cond, None)

def kalimba_maxim_decode_doloop_b(instruction, op):
    (k16, _, regc) = kalimba_maxim_decode_b(instruction)

    return KalimbaControlFlow(op, k16, KalimbaCond.Always, None)

@dataclass(unsafe_hash=True)
class KalimbaExtraStackAdjust:
    '''
    RegC = SP/FP OP RegOff
    '''
    op: Union[Literal[KalimbaOp.ADD], Literal[KalimbaOp.SUB]]
    c: KalimbaReg
    rst: Union[Literal[KalimbaBank3Reg.SP], Literal[KalimbaBank3Reg.FP]]
    off: KalimbaBank1Reg

    def __str__(self):
        return f'{self.c} = {self.rst} {binop_symbols[self.op]} {self.off}'

@dataclass(unsafe_hash=True)
class KalimbaStackOp:
    '''
    PUSH/POP
    '''
    op: KalimbaOp
    reg: List[KalimbaReg]
    cond: KalimbaCond
    adj: Optional[KalimbaExtraStackAdjust]
    mem: Optional[KalimbaIndexedMemAccess]
    def __str__(self):
        m = f', {self.mem}' if self.mem else ''
        adj = f', {self.adj}' if self.adj else ''
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        reg = str(self.reg[0]) if len(self.reg) == 1 else f'<{', '.join(map(str, self.reg))}>'
        return f'KalimbaStackOp("{cond}{self.op.name.lower()} {reg}{adj}{m}")'

def kalimba_maxim_decode_stack_a(instruction, op):
    bankc = reg_bank_lut[get_bits(instruction, 16, 2) + 1]
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction, KalimbaBank1Reg, KalimbaBank1Reg, bankc)

    if op in [KalimbaOp.STORES, KalimbaOp.LOADS]:
        regb = KalimbaBank1Reg(get_bits(instruction, 4, 4))
        return KalimbaOffsetMemAccess(op, regc, KalimbaBank3Reg.FP, regb, cond, mem)
    else:
        return KalimbaStackOp(op, [regc], cond, None, mem)

def kalimba_maxim_decode_stack_adj_a(instruction, op):
    (cond, mem, rega, regb, regc) = kalimba_maxim_decode_a(instruction)

    if (rega.value & 0b100) == 0b100:
        rega = KalimbaBank3Reg.FP
    else:
        rega = KalimbaBank3Reg.SP

    if regc is KalimbaBank1Reg.Null:
        regc = rega;

    return KalimbaBinOp(op, regc, rega, regb, cond, mem)

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
    cond: KalimbaCond
    sub: bool
    def __str__(self):
        m = self.sel.name[2:]
        cond = '' if self.cond == KalimbaCond.Always else f'if {self.cond.name} '
        op = '-' if self.sub else '+'

        if self.op in [KalimbaOp.LOADW]:
            return f'KalimbaOffsetMemAccess("{cond}{self.c} = {m}[{self.a} {op} {self.b}]")'
        elif self.op in [KalimbaOp.STOREW]:
            return f'KalimbaOffsetMemAccess("{cond}{m}[{self.a} {op} {self.b}] = {self.c}")'
        else:
            raise ValueError(f'invalid mem op: {self.op}')

bank_bit_lut = [KalimbaBank1Reg, KalimbaBank2Reg]

def kalimba_maxim_decode_subword_a(instruction, op):
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

def kalimba_maxim_decode_subword_b(instruction, op):
    sel = KalimbaSubWordMem(get_bits(instruction, 13, 3))
    sub = get_bits(instruction, 9, 1) == 1
    banka = bank_bit_lut[get_bits(instruction, 11, 1)]
    bankc = bank_bit_lut[get_bits(instruction, 12, 1)]
    (k16, regb, regc) = kalimba_maxim_decode_b(instruction, banka, bankc)

    if sel in [KalimbaSubWordMem.S_M, KalimbaSubWordMem.S_MB, KalimbaSubWordMem.S_MH]:
        op = KalimbaOp.STOREW
    else:
        op = KalimbaOp.LOADW

    return KalimbaSubWordMemAccess(op, sel, regc, rega, regb, cond, sub)

@dataclass(unsafe_hash=True)
class KalimbaPrefix:
    op: KalimbaOp
    const: int
    def __str__(self):
        return f'KalimbaPrefix(0x{self.const:x})'

def kalimba_maxim_decode_prefix(instruction, op):
    prefix = get_bits(instruction, 0, 21)
    return KalimbaPrefix(op, prefix)

type KalimbaInstruction = Union[KalimbaUnOp, KalimbaBinOp, KalimbaFusedMultiplyAddSub, KalimbaOffsetMemAccess, KalimbaControlFlow, KalimbaSubWordMemAccess]

# mask, value, operation, decode
maxim_ops_lut: List[Union[Tuple[int, int, KalimbaOp, Callable[[int, KalimbaOp], KalimbaInstruction]], Tuple[int, int, KalimbaOp]]] = [
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
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00001000_00000000_00000000, KalimbaOp.STORES, kalimba_maxim_decode_stack_a),
    (0b111111_11_00001100_00000000_00000000, 0b111_100_00_00001100_00000000_00000000, KalimbaOp.LOADS,  kalimba_maxim_decode_stack_a),
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
    # TODO: stack
    #(0b111111_11_00001111_00000000_00000000, 0b111_100_01_00000011_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_b),
    #(0b111111_11_00001111_00000000_00000000, 0b111_100_01_00000111_00000000_00000000, KalimbaOp.ADD, kalimba_maxim_decode_stack_adj_b),
    #(0b111111_11_00001100_00000000_00000000, 0b111_100_01_00000000_00000000_00000000, KalimbaOp.PUSH,   kalimba_maxim_decode_stack_b),
    #(0b111111_11_00001100_00000000_00000000, 0b111_100_01_00000100_00000000_00000000, KalimbaOp.POP,    kalimba_maxim_decode_stack_b),
    #(0b111111_11_00001100_00000000_00000000, 0b111_100_01_00001000_00000000_00000000, KalimbaOp.STORES, kalimba_maxim_decode_stack_b),
    #(0b111111_11_00001100_00000000_00000000, 0b111_100_01_00001100_00000000_00000000, KalimbaOp.LOADS,  kalimba_maxim_decode_stack_b),

    # Type C
    #(),

    # Subword A
    (0b11111111_00000000_00000000_00000000, 0b11110100_00000000_00000000_00000000, KalimbaOp.LOADW, kalimba_maxim_decode_subword_a),
    # Subword B
    (0b11111111_00000000_00000000_00000000, 0b11110101_00000000_00000000_00000000, KalimbaOp.LOADW, kalimba_maxim_decode_subword_b),
    # Prefix
    (0b11111111_11100000_00000000_00000000, 0b11111101_00000000_00000000_00000000, KalimbaOp.PREFIX, kalimba_maxim_decode_prefix),
]

def kalimba_maxim_lookup_op(instruction: int) -> Union[KalimbaInstruction, str]:
    for (mask, value, opcode, *other) in maxim_ops_lut:
        if (instruction & mask) == value:
            return opcode.name if not other else other[0](instruction, opcode)
    raise ValueError(f'invalid instruction 0b{instruction:032b}')

if __name__ == '__main__':
    #print(kalimba_maxim_lookup_op(0x5812002f))# 2f 00 12 58 | I1 = I2 + r0;
    #print(kalimba_maxim_lookup_op(0x0034005f))# 5f 00 34 00 | r1 = r2 + r3;
    #print(kalimba_maxim_lookup_op(0x0434005f))# 5f 00 34 04 | r1 = r2 + r3 + Carry;
    #print(kalimba_maxim_lookup_op(0x0035004f))# 4f 00 35 00 | r1 = r3 + r2;
    #print(kalimba_maxim_lookup_op(0x2034005f))# 5f 00 34 20 | r1 = r2 - r3;
    #print(kalimba_maxim_lookup_op(0x2035004f))# 4f 00 35 20 | r1 = r3 - r2;
    #print(kalimba_maxim_lookup_op(0x2435004f))# 4f 00 35 24 | r1 = r3 - r2 - Borrow;
    #print(kalimba_maxim_lookup_op(0x1834005f))# 5f 00 34 18 | M[r1] = r2 + r3;
    #print(kalimba_maxim_lookup_op(0x18340056))# 56 00 34 18 | if V M[r1] = r2 + r3;
    #print(kalimba_maxim_lookup_op(0x18342456))# 56 24 34 18 | if V M[r1] = r2 + r3, r0 = M[I1,M0];
    #print(kalimba_maxim_lookup_op(0xe434454f))# 4f 45 34 e4 | r1 = ABS r2, r2 = M[I1,M1];
    #print(kalimba_maxim_lookup_op(0xe434ba4f))# 4f ba 34 e4 | r1 = ABS r2, M[I2,M2] = r1;
    #print(kalimba_maxim_lookup_op(0x8834005f))# 5f 00 34 88 | r1 = r2 XOR r3;
    #print(kalimba_maxim_lookup_op(0x8cea00b1))# b1 00 ea 8c | if Z rMAC = r8 LSHIFT r9;
    #print(kalimba_maxim_lookup_op(0x90e70090))# 90 00 e7 90 | if EQ rMAC = r5 ASHIFT r7;
    #print(kalimba_maxim_lookup_op(0x88170090))# 90 00 17 88 | if EQ rMAC = r5 XOR r7;
    #print(kalimba_maxim_lookup_op(0x00170090))# 90 00 17 00 | if EQ rMAC = r5 + r7;
    #print(kalimba_maxim_lookup_op(0x88e70090))# 90 00 e7 88 | if EQ rFlags = r5 XOR r7;
    #print(kalimba_maxim_lookup_op(0x9434005f))# 5f 00 34 94 | r1 = r2 * r3 (frac);
    #print(kalimba_maxim_lookup_op(0x9834005f))# 5f 00 34 98 | r1 = r2 * r3 (int);
    #print(kalimba_maxim_lookup_op(0x9c34005f))# 5f 00 34 9c | r1 = r2 * r3 (int) (sat);
    #print(kalimba_maxim_lookup_op(0x9cf4005f))# 5f 00 f4 9c | rMACB = r2 * r3 (int) (sat);
    #print(kalimba_maxim_lookup_op(0xac23504f))# 4f 50 23 ac | r0 = r0 + r1 * r2 (SS), r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xa4c3504f))# 4f 50 c3 a4 | rMACB = rMACB + r1 * r2, r0 = r1 - rMACB, r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xacc5906f))# 6f 90 c5 ac | rMACB = rMACB + r3 * r4, r0 = r2 - rMACB, M[I0,M0] = rMAC;
    #print(kalimba_maxim_lookup_op(0xa483504f))# 4f 50 83 a4 | rMACB = rMACB + r1 * r2, r0 = r1 - rMAC, r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xac85906f))# 6f 90 85 ac | rMACB = rMACB + r3 * r4, r0 = r2 - rMAC, M[I0,M0] = rMAC;
    #print(kalimba_maxim_lookup_op(0xb0d3504f))# 4f 50 d3 b0 | rMAC = rMAC - r1 * r2, r0 = r1 + rMACB, r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xb8d5906f))# 6f 90 d5 b8 | rMAC = rMAC - r3 * r4, r0 = r2 + rMACB, M[I0,M0] = rMAC;
    #print(kalimba_maxim_lookup_op(0xa803504e))# 4e 50 03 a8 | if USERDEF rMACB = rMACB + r1 * r2 (SU), r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xc4d3504f))# 4f 50 d3 c4 | rMAC = r1 * r2, r0 = r1 - rMACB, r3 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xcc85906f))# 6f 90 85 cc | rMACB = r3 * r4, r0 = r2 - rMAC, M[I0,M0] = rMAC;
    #print(kalimba_maxim_lookup_op(0x581200ef))# ef 00 12 58 | I1 = I2 + FP;
    #print(kalimba_maxim_lookup_op(0x541e002f))# 2f 00 1e 54 | I1 = FP + I2;
    #print(kalimba_maxim_lookup_op(0x4ce1002f))# 2f 00 e1 4c | rFlags = I1 + I2;
    #print(kalimba_maxim_lookup_op(0xd02140fe))# fe 40 21 d0 | if USERDEF r0 = M[rMAC + rMACB], r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xd441a1fe))# fe a1 41 d4 | if USERDEF M[rMAC + rMACB] = r2, M[I0,M1] = r0;
    #print(kalimba_maxim_lookup_op(0xe434002f))# 2f 00 34 e4 | r1 = r2 + 1;
    #print(kalimba_maxim_lookup_op(0xe434003f))# 3f 00 34 e4 | r1 = r2 - 1;
    #print(kalimba_maxim_lookup_op(0xe434004f))# 4f 00 34 e4 | r1 = ABS r2;
    #print(kalimba_maxim_lookup_op(0xe434005f))# 5f 00 34 e4 | r1 = MIN r2;
    #print(kalimba_maxim_lookup_op(0xe434006f))# 6f 00 34 e4 | r1 = MAX r2;
    #print(kalimba_maxim_lookup_op(0xe434007f))# 7f 00 34 e4 | r1 = TWOBITCOUNT r2;
    #print(kalimba_maxim_lookup_op(0xe434008f))# 8f 00 34 e4 | r1 = MOD24 r2;
    #print(kalimba_maxim_lookup_op(0xe43400af))# af 00 34 e4 | r1 = r2 + 2;
    #print(kalimba_maxim_lookup_op(0xe43400bf))# bf 00 34 e4 | r1 = r2 + 4;
    #print(kalimba_maxim_lookup_op(0xe43400cf))# cf 00 34 e4 | r1 = r2 - 2;
    #print(kalimba_maxim_lookup_op(0xe43400df))# df 00 34 e4 | r1 = r2 - 4;
    #print(kalimba_maxim_lookup_op(0xe43400ef))# ef 00 34 e4 | r1 = SE8 r2;
    #print(kalimba_maxim_lookup_op(0xe43400ff))# ff 00 34 e4 | r1 = SE16 r2;
    #print(kalimba_maxim_lookup_op(0xd823000f))# 0f 00 23 d8 | r0 = SIGNDET r1;
    #print(kalimba_maxim_lookup_op(0xdc0d000f))# 0f 00 0d dc | rts; // special case of jump
    #print(kalimba_maxim_lookup_op(0xdc0e000f))# 0f 00 0e dc | rti; // special case of jump
    #print(kalimba_maxim_lookup_op(0xdc09000e))# 0e 00 09 dc | if USERDEF jump r7;
    #print(kalimba_maxim_lookup_op(0xe00c0000))# 00 00 0c e0 | if EQ call r10;
    #print(kalimba_maxim_lookup_op(0xe00ca800))# 00 a8 0c e0 | if EQ call r10, M[I2,M0] = r0;
    #print(kalimba_maxim_lookup_op(0xdc0eaf01))# 01 af 0e dc | if NE rti, M[I3,M3] = r0;
    #print(kalimba_maxim_lookup_op(0xdc0daf01))# 01 af 0d dc | if NE rts, M[I3,M3] = r0;
    #print(kalimba_maxim_lookup_op(0xf0904006))# 06 40 90 f0 | if V push r7, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0d04006))# 06 40 d0 f0 | if V push rLink, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0814006))# 06 40 81 f0 | if V push M0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0a24006))# 06 40 a2 f0 | if V push B0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0624006))# 06 40 62 f0 | if V push DivRemainder, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0934025))# 25 40 93 f0 | if POS r7 = SP + r0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0034025))# 25 40 03 f0 | if POS SP = SP + r0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0974025))# 25 40 97 f0 | if POS r7 = FP + r0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf0074025))# 25 40 07 f0 | if POS FP = FP + r0, r2 = M[I0,M0];
    #print(kalimba_maxim_lookup_op(0xf429cc1f))# 1f cc 29 f4 | MH[M1 + I1] = r0;
    #print(kalimba_maxim_lookup_op(0xf429ec1f))# 1f ec 29 f4 | M[M1 + I1] = r0;
    #print(kalimba_maxim_lookup_op(0xf429ac1f))# 1f ac 29 f4 | MB[M1 + I1] = r0;
    #print(kalimba_maxim_lookup_op(0xf4f2640f))# 0f 64 f2 f4 | rMACB = MHU[r0 + I0];
    #print(kalimba_maxim_lookup_op(0xf4f2040f))# 0f 04 f2 f4 | rMACB = MBS[r0 + I0];
    #print(kalimba_maxim_lookup_op(0xf4f2840f))# 0f 84 f2 f4 | rMACB = M[r0 + I0];
    #print(kalimba_maxim_lookup_op(0xe4b000bf))# bf 00 b0 e4 | r9 = Null + 4; // Type A (ADD4)

    #print(kalimba_maxim_lookup_op(0xfd000000))# 00 00 00 fd | r9 = r1 + 32768;
    #print(kalimba_maxim_lookup_op(0x01b38000))# 00 80 b3 01
    #print(kalimba_maxim_lookup_op(0x01b00004))# 04 00 b0 01 | r9 = Null + 4; // Type B
    #print(kalimba_maxim_lookup_op(0x01b00009))# 09 00 b0 01 | r9 = Null + 9;
    #print(kalimba_maxim_lookup_op(0x01b07fff))# ff 7f b0 01 | r9 = Null + 32767;
    #print(kalimba_maxim_lookup_op(0x01b37fff))# ff 7f b3 01 | r9 = r1 + 32767;
    #print(kalimba_maxim_lookup_op(0x21b37fff))# ff 7f b3 21 | r9 = r1 - 32767;

    print(kalimba_maxim_lookup_op(0x053b0007))# 07 00 3b 05 | r1 = r9 + 7 + Carry;
    print(kalimba_maxim_lookup_op(0x0d3b0007))# 07 00 3b 0d | r1 = r9 + M[0x7] + Carry;
    print(kalimba_maxim_lookup_op(0x153b0007))# 07 00 3b 15 | r1 = M[r9] + 7 + Carry;
    print(kalimba_maxim_lookup_op(0x1d3b0007))# 07 00 3b 1d | M[0x7] = r1 + r9 + Carry;
    print(kalimba_maxim_lookup_op(0x1d3b0009))# 09 00 3b 1d | M[0x9] = r1 + r9 + Carry;
    print(kalimba_maxim_lookup_op(0x3d3b0009))# 09 00 3b 3d | M[0x9] = r1 - r9 - Borrow;
    print(kalimba_maxim_lookup_op(0x15b30009))# 09 00 b3 15 | r9 = M[r1] + 9 + Carry;
    print(kalimba_maxim_lookup_op(0x0db30009))# 09 00 b3 0d | r9 = r1 + M[0x9] + Carry;
    print(kalimba_maxim_lookup_op(0x59120007))# 07 00 12 59 | I1 = I2 + 7;
    print(kalimba_maxim_lookup_op(0x7912fff9))# f9 ff 12 79 | I1 = I2 - -7;
    print(kalimba_maxim_lookup_op(0x7d12fff9))# f9 ff 12 7d | I1 = -7 - I2;
    print(kalimba_maxim_lookup_op(0x511e0007))# 07 00 1e 51 | I1 = FP + 7;
    print(kalimba_maxim_lookup_op(0x892b0005))# 05 00 2b 89 | r0 = r9 XOR 0x5;
    print(kalimba_maxim_lookup_op(0x912b0005))# 05 00 2b 91 | r0 = r9 ASHIFT 5;
    print(kalimba_maxim_lookup_op(0x912b00fb))# fb 00 2b 91 | r0 = r9 ASHIFT -5;
    print(kalimba_maxim_lookup_op(0x8d2b00fb))# fb 00 2b 8d | r0 = r9 LSHIFT -5;
    print(kalimba_maxim_lookup_op(0x8deb0005))# 05 00 eb 8d | rMAC = r9 LSHIFT 5 (MI);
    print(kalimba_maxim_lookup_op(0x8d1b0005))# 05 00 1b 8d | rMAC = r9 LSHIFT 5 (56bit);
    print(kalimba_maxim_lookup_op(0x8dfb0005))# 05 00 fb 8d | rMACB = r9 LSHIFT 5 (56bit);
    print(kalimba_maxim_lookup_op(0x8ceb002f))# 2f 00 eb 8c | rMAC = r9 LSHIFT r0;
    print(kalimba_maxim_lookup_op(0x8c1b002f))# 2f 00 1b 8c | rMAC = r9 LSHIFT r0 (56bit);
    print(kalimba_maxim_lookup_op(0x8cfb002f))# 2f 00 fb 8c | rMACB = r9 LSHIFT r0 (56bit);
    print(kalimba_maxim_lookup_op(0x91f90507))# 07 05 f9 91 | rMACB0 = r7 ASHIFT 7;
    print(kalimba_maxim_lookup_op(0x91f90407))# 07 04 f9 91 | rMACB12 = r7 ASHIFT 7;
    print(kalimba_maxim_lookup_op(0x91f906f9))# f9 06 f9 91 | rMACB2 = r7 ASHIFT -7;
    print(kalimba_maxim_lookup_op(0x91f106f9))# f9 06 f1 91 | rMACB2 = rMAC ASHIFT -7;
    print(kalimba_maxim_lookup_op(0x91f102f9))# f9 02 f1 91 | rMACB = rMAC ASHIFT -7 (HI);
    print(kalimba_maxim_lookup_op(0x91e100f9))# f9 00 e1 91 | rMAC = rMAC ASHIFT -7 (MI);
    print(kalimba_maxim_lookup_op(0x91f101f9))# f9 01 f1 91 | rMACB = rMAC ASHIFT -7 (LO);
    print(kalimba_maxim_lookup_op(0x91f100f9))# f9 00 f1 91 | rMACB = rMAC ASHIFT -7 (56bit);

    print(kalimba_maxim_lookup_op(0x991c0007))# 07 00 1c 99 | rMAC = r10 * 7 (int);
    print(kalimba_maxim_lookup_op(0x9927fff9))# f9 ff 27 99 | r0 = r5 * -7 (int);
    print(kalimba_maxim_lookup_op(0x9d27fff9))# f9 ff 27 9d | r0 = r5 * -7 (int) (sat);
    print(kalimba_maxim_lookup_op(0x95274000))# 00 40 27 95 | r0 = r5 * 0.5 (frac);

    #print(kalimba_maxim_lookup_op(0xdde0fffe))# fe ff e0 dd | if USERDEF jump BRANCH1;
    #print(kalimba_maxim_lookup_op(0xe1000004))# 04 00 00 e1 | if EQ call BRANCH2;
    #print(kalimba_maxim_lookup_op(0x03000000))# Type C
    #print(kalimba_maxim_lookup_op(0x23000000))

