import enum
from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union
stack_instructions = ['pushm', 'pushc','push', 'popm', 'popm_rts', 'pop']
rw_data_instructions = [
'=MBS','=MBU','=MHS','=MHU','=M',
'MB','MH','M'
]
al_instructions = [
    'AND',
    'OR',
    'XOR',
    '-',#SUB
    '+',#ADD
    'LSHIFT',
    'ASHIFT',
    '*',#IMULT
    'SE8',
    'SE16',
    '/',#Div
    'mv',#no standard
]
program_flow_instructions = [
    'call',
    'call(m)',
    'jump',
    'jump(m)',
    'do',
    'do(m)'
]
class kalimba_minim_instr_type(enum.IntEnum):
    TYPE_A = 0
    TYPE_B = 1
    TYPE_C = 2
    TYPE_NO_REGB_K = 3
class stack_instructions_param():
    def __init__(self):
        self.sp_adjust = 0
        self.reg_list = []
class rw_data_instructions_param():
    def __init__(self):
        self.bits = 32
        self.add = True
class al_instructions_param():
    def __init__(self):
        self.bits = 32
        self.use_carry = False
class program_flow_instructions_param():
    def __init__(self):
        self.cond = ''
class kalimba_minim_instr():
    param: Union[stack_instructions_param,rw_data_instructions_param,al_instructions_param,program_flow_instructions_param]
    instr_type:kalimba_minim_instr_type
    def __init__(self):
        self.instr_type = kalimba_minim_instr_type.TYPE_A
        self.regc = ''
        self.rega = ''
        self.op = ''
        self.regb_k = ''
        self.param = None
        self.length = 0
        self.is_insert32 = False