from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union

from binaryninja.architecture import Architecture, InstructionInfo, RegisterInfo
from binaryninja.lowlevelil import LowLevelILFunction
from binaryninja.function import InstructionTextToken
from binaryninja.enums import InstructionTextTokenType,FlagRole,BranchType,LowLevelILFlagCondition

from binaryninja.log import log_info

from .k_instr import *
from .k_dis import get_disassembly_description

class KALIMBA(Architecture):
    name = 'KALIMBA'
    address_size = 4
    default_int_size = 4
    instr_alignment = 2
    max_instr_length = 8
    regs = {
        #bank1
        'Null': RegisterInfo('Null', 9),
        'rMAC': RegisterInfo('rMAC', 9),

        'r0': RegisterInfo('r0', 4),
        'r1': RegisterInfo('r1', 4),
        'r2': RegisterInfo('r2', 4),
        'r3': RegisterInfo('r3', 4),
        'r4': RegisterInfo('r4', 4),
        'r5': RegisterInfo('r5', 4),
        'r6': RegisterInfo('r6', 4),
        'r7': RegisterInfo('r7', 4),
        'r8': RegisterInfo('r8', 4),
        'r9': RegisterInfo('r9', 4),
        'r10': RegisterInfo('r10', 4),

        'rLink': RegisterInfo('rLink', 4),
        'rFlags': RegisterInfo('rFlags', 4),

        'rMACB': RegisterInfo('rMACB', 9),
        #bank2
        'I0': RegisterInfo('I0', 4),
        'I1': RegisterInfo('I1', 4),
        'I2': RegisterInfo('I2', 4),
        'I3': RegisterInfo('I3', 4),
        'I4': RegisterInfo('I4', 4),
        'I5': RegisterInfo('I5', 4),
        'I6': RegisterInfo('I6', 4),
        'I7': RegisterInfo('I7', 4),

        'M0': RegisterInfo('M0', 4),
        'M1': RegisterInfo('M1', 4),
        'M2': RegisterInfo('M2', 4),
        'M3': RegisterInfo('M3', 4),

        'L0': RegisterInfo('L0', 4),
        'L1': RegisterInfo('L1', 4),
        'L4': RegisterInfo('L4', 4),
        'L5': RegisterInfo('L5', 4),
        #bank3
        'rMAC2': RegisterInfo('rMAC', 1, 64),
        'rMAC1': RegisterInfo('rMAC', 4, 32),
        'rMAC0': RegisterInfo('rMAC', 4, 0),

        'DoLoopStart': RegisterInfo('DoLoopStart', 4),
        'DoLoopEnd': RegisterInfo('DoLoopEnd', 4),
        'DivResult': RegisterInfo('DivResult', 4),
        'DivRemainder': RegisterInfo('DivRemainder', 4),

        'rMACB2': RegisterInfo('rMACB', 1, 64),
        'rMACB1': RegisterInfo('rMACB', 4, 32),
        'rMACB0': RegisterInfo('rMACB', 4, 0),

        'B0': RegisterInfo('B0', 4),
        'B1': RegisterInfo('B1', 4),
        'B4': RegisterInfo('B4', 4),
        'B5': RegisterInfo('B5', 4),

        'FP': RegisterInfo('FP', 4),
        'SP': RegisterInfo('SP', 4),
    }
    '''
    flag_roles = {
        'Z': FlagRole.ZeroFlagRole,
        'C': FlagRole.CarryFlagRole,
        'N': FlagRole.SpecialFlagRole,
        'V': FlagRole.OverflowFlagRole,
    }
    '''
    stack_pointer = 'SP'
    link_reg = 'rLink'
    minim_offset = 0#0x180
    
    def get_instruction_info(self, data:bytes, addr:int) -> Optional[InstructionInfo]:
        result = InstructionInfo()
        description = get_disassembly_description(data, addr)
        if not description:
            result.length = 2
            return result
        result.length = description.length
        if True:
            if description.op in program_flow_instructions:
                if description.instr_type == kalimba_minim_instr_type.TYPE_B:
                    if 'jump'in description.op:
                        if description.param and description.param.cond != '' and description.param.cond != 'Always':
                            result.add_branch(BranchType.TrueBranch, description.regb_k + addr)
                            result.add_branch(BranchType.FalseBranch, addr + description.length)
                        else:
                            result.add_branch(BranchType.UnconditionalBranch, description.regb_k + addr)
                    elif 'call'in description.op:
                        result.add_branch(BranchType.CallDestination, description.regb_k + addr)
                else:
                    if 'call'in description.op and 'Null' in description.regc:
                        result.add_branch(BranchType.FunctionReturn)
        return result
    def _padding(self, s=''):
        return InstructionTextToken(InstructionTextTokenType.TextToken, f'{s} ')

    def get_instruction_text(self, data: bytes, addr: int) -> Optional[Tuple[List[InstructionTextToken], int]]:
        description = get_disassembly_description(data, addr)
        ops = []
        '''
        if isinstance(description.regb_k, int) and addr < 0x80010610 and description.regb_k + addr >= 0x80010610 and description.regb_k + addr < 0x81000000:
            log_info(f'addr {hex(addr)} error')
        '''
        if not description:
            return None
        if description.op == '':
            return ops, description.length
        if description.is_insert32:
            ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'i32'))
            ops.append(self._padding())
        if description.op in stack_instructions:
            if description.op == 'pushm':
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'pushm'))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '<'))
                for reg in description.param.reg_list:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, reg))
                    ops.append(self._padding(','))
                ops.pop()
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '>'))
                if description.param.sp_adjust > 0:
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '='))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '+'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(description.param.sp_adjust), description.param.sp_adjust))
            elif description.op == 'pushc':
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'push'))
                ops.append(self._padding())
                if description.regc != '':
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regc))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '+'))
                    ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(description.regb_k), description.regb_k))
            elif description.op == 'popm':
                if description.param.sp_adjust < 0:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '='))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '-'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(-description.param.sp_adjust), description.param.sp_adjust))
                    ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'popm'))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '<'))
                for reg in description.param.reg_list:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, reg))
                    ops.append(self._padding(','))
                ops.pop()
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '>'))
            elif description.op == 'popm_rts':
                if description.param.sp_adjust < 0:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '='))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, 'SP'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '-'))
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(-description.param.sp_adjust), description.param.sp_adjust))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'popm_rts'))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '<'))
                for reg in description.param.reg_list:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, reg))
                    ops.append(self._padding(','))
                ops.pop()
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '>'))
        elif description.op in rw_data_instructions:            
            if '=' in description.op:
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regc))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '='))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, description.op[1:]))
                ops.append(InstructionTextToken(InstructionTextTokenType.BeginMemoryOperandToken, '['))
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.rega))
                ops.append(self._padding())
                if description.param and description.param.add == False:
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '-'))
                else:
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '+'))
                ops.append(self._padding())
                if description.instr_type == kalimba_minim_instr_type.TYPE_A:

                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regb_k))
                else:
                    ops.append(InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(description.regb_k), description.regb_k))
                ops.append(InstructionTextToken(InstructionTextTokenType.EndMemoryOperandToken, ']'))
            else:
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, description.op))
                ops.append(InstructionTextToken(InstructionTextTokenType.BeginMemoryOperandToken, '['))
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.rega))
                ops.append(self._padding())
                if description.param and description.param.add == False:
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '-'))
                else:
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, '+'))
                ops.append(self._padding())
                if description.instr_type == kalimba_minim_instr_type.TYPE_A:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regb_k))
                else:
                    ops.append(InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(description.regb_k), description.regb_k))
                ops.append(InstructionTextToken(InstructionTextTokenType.EndMemoryOperandToken, ']'))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '='))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regc))
        elif description.op in al_instructions:
            ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regc))
            ops.append(self._padding())
            ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, '='))
            ops.append(self._padding())
            if description.param and description.param.shift_reverse:
                if description.instr_type == kalimba_minim_instr_type.TYPE_B:
                    ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(description.regb_k), description.regb_k))
                    ops.append(self._padding())
                else:
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regb_k))
                    ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, description.op))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.rega))
            else:
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.rega))
                if description.op != 'mv':
                    ops.append(self._padding())
                    ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, description.op))
                    ops.append(self._padding())
                if description.instr_type == kalimba_minim_instr_type.TYPE_B:
                    ops.append(InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(description.regb_k), description.regb_k))
                else:
                    
                    ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regb_k))

        elif description.op in program_flow_instructions:
            if description.param and description.param.cond != '':
                ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, 'if'))
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, description.param.cond))
                ops.append(self._padding())
            ops.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, description.op))
            if description.instr_type == kalimba_minim_instr_type.TYPE_B:
                ops.append(self._padding())
                ops.append(InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(description.regb_k+addr) ,description.regb_k+addr))
            elif description.regc != '':
                ops.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, description.regc))
        elif description.is_insert32:
            ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, description.op))

        return ops, description.length # len of instruction

    def get_instruction_low_level_il(self, data: bytes, addr: int, il: LowLevelILFunction) -> Optional[int]:
        #tmp = il.reg(1, 'A')
        #tmp = il.add_carry
        #tmp = il.set_reg(1, 'A', tmp)
        return None
