from typing import Callable, List, Type, Optional, Dict, Tuple, NewType, Union

from binaryninja.architecture import Architecture, InstructionInfo, RegisterInfo
from binaryninja.lowlevelil import LowLevelILFunction
from binaryninja.function import InstructionTextToken
from binaryninja.enums import InstructionTextTokenType,FlagRole,BranchType,LowLevelILFlagCondition

from binaryninja.log import log_info

try:
    from k_instr_new import *
    from k_minim import kalimba_minim_decode
except ImportError:
    from .k_instr_new import *
    from .k_minim import kalimba_minim_decode

class KALIMBA(Architecture):
    name = 'KALIMBA'
    address_size = 4
    default_int_size = 4
    instr_alignment = 2
    max_instr_length = 8
    regs = {
        #bank1
        KalimbaBank1Reg.Null.name: RegisterInfo(KalimbaBank1Reg.Null.name, 9),
        KalimbaBank1Reg.rMAC.name: RegisterInfo(KalimbaBank1Reg.rMAC.name, 9),

        KalimbaBank1Reg.r0.name: RegisterInfo(KalimbaBank1Reg.r0.name, 4),
        KalimbaBank1Reg.r1.name: RegisterInfo(KalimbaBank1Reg.r1.name, 4),
        KalimbaBank1Reg.r2.name: RegisterInfo(KalimbaBank1Reg.r2.name, 4),
        KalimbaBank1Reg.r3.name: RegisterInfo(KalimbaBank1Reg.r3.name, 4),
        KalimbaBank1Reg.r4.name: RegisterInfo(KalimbaBank1Reg.r4.name, 4),
        KalimbaBank1Reg.r5.name: RegisterInfo(KalimbaBank1Reg.r5.name, 4),
        KalimbaBank1Reg.r6.name: RegisterInfo(KalimbaBank1Reg.r6.name, 4),
        KalimbaBank1Reg.r7.name: RegisterInfo(KalimbaBank1Reg.r7.name, 4),
        KalimbaBank1Reg.r8.name: RegisterInfo(KalimbaBank1Reg.r8.name, 4),
        KalimbaBank1Reg.r9.name: RegisterInfo(KalimbaBank1Reg.r9.name, 4),
        KalimbaBank1Reg.r10.name: RegisterInfo(KalimbaBank1Reg.r10.name, 4),

        KalimbaBank1Reg.rLink.name: RegisterInfo(KalimbaBank1Reg.rLink.name, 4),
        KalimbaBank1Reg.rFlags.name: RegisterInfo(KalimbaBank1Reg.rFlags.name, 4),

        KalimbaBank1Reg.rMACB.name: RegisterInfo(KalimbaBank1Reg.rMACB.name, 9),
        #bank2
        KalimbaBank2Reg.I0.name: RegisterInfo(KalimbaBank2Reg.I0.name, 4),
        KalimbaBank2Reg.I1.name: RegisterInfo(KalimbaBank2Reg.I1.name, 4),
        KalimbaBank2Reg.I2.name: RegisterInfo(KalimbaBank2Reg.I2.name, 4),
        KalimbaBank2Reg.I3.name: RegisterInfo(KalimbaBank2Reg.I3.name, 4),
        KalimbaBank2Reg.I4.name: RegisterInfo(KalimbaBank2Reg.I4.name, 4),
        KalimbaBank2Reg.I5.name: RegisterInfo(KalimbaBank2Reg.I5.name, 4),
        KalimbaBank2Reg.I6.name: RegisterInfo(KalimbaBank2Reg.I6.name, 4),
        KalimbaBank2Reg.I7.name: RegisterInfo(KalimbaBank2Reg.I7.name, 4),

        KalimbaBank2Reg.M0.name: RegisterInfo(KalimbaBank2Reg.M0.name, 4),
        KalimbaBank2Reg.M1.name: RegisterInfo(KalimbaBank2Reg.M1.name, 4),
        KalimbaBank2Reg.M2.name: RegisterInfo(KalimbaBank2Reg.M2.name, 4),
        KalimbaBank2Reg.M3.name: RegisterInfo(KalimbaBank2Reg.M3.name, 4),

        KalimbaBank2Reg.L0.name: RegisterInfo(KalimbaBank2Reg.L0.name, 4),
        KalimbaBank2Reg.L1.name: RegisterInfo(KalimbaBank2Reg.L1.name, 4),
        KalimbaBank2Reg.L4.name: RegisterInfo(KalimbaBank2Reg.L4.name, 4),
        KalimbaBank2Reg.L5.name: RegisterInfo(KalimbaBank2Reg.L5.name, 4),
        #bank3
        KalimbaBank3Reg.rMAC2.name: RegisterInfo(KalimbaBank1Reg.rMAC.name, 1, 64),
        KalimbaBank3Reg.rMAC1.name: RegisterInfo(KalimbaBank1Reg.rMAC.name, 4, 32),
        KalimbaBank3Reg.rMAC0.name: RegisterInfo(KalimbaBank1Reg.rMAC.name, 4, 0),

        KalimbaBank3Reg.DoLoopStart.name: RegisterInfo(KalimbaBank3Reg.DoLoopStart.name, 4),
        KalimbaBank3Reg.DoLoopEnd.name: RegisterInfo(KalimbaBank3Reg.DoLoopEnd.name, 4),
        KalimbaBank3Reg.DivResult.name: RegisterInfo(KalimbaBank3Reg.DivResult.name, 4),
        KalimbaBank3Reg.DivRemainder.name: RegisterInfo(KalimbaBank3Reg.DivRemainder.name, 4),

        KalimbaBank3Reg.rMACB2.name: RegisterInfo(KalimbaBank1Reg.rMACB.name, 1, 64),
        KalimbaBank3Reg.rMACB1.name: RegisterInfo(KalimbaBank1Reg.rMACB.name, 4, 32),
        KalimbaBank3Reg.rMACB0.name: RegisterInfo(KalimbaBank1Reg.rMACB.name, 4, 0),

        KalimbaBank3Reg.B0.name: RegisterInfo(KalimbaBank3Reg.B0.name, 4),
        KalimbaBank3Reg.B1.name: RegisterInfo(KalimbaBank3Reg.B1.name, 4),
        KalimbaBank3Reg.B4.name: RegisterInfo(KalimbaBank3Reg.B4.name, 4),
        KalimbaBank3Reg.B5.name: RegisterInfo(KalimbaBank3Reg.B5.name, 4),

        KalimbaBank3Reg.FP.name: RegisterInfo(KalimbaBank3Reg.FP.name, 4),
        KalimbaBank3Reg.SP.name: RegisterInfo(KalimbaBank3Reg.SP.name, 4),
    }
    #Actually used
    flags = [
        KalimbaFlags.Z.name,
        KalimbaFlags.N.name,
        KalimbaFlags.V.name,
        KalimbaFlags.C.name,
        KalimbaFlags.UD.name
    ]
    flag_roles = {
        KalimbaFlags.Z.name: FlagRole.ZeroFlagRole,
        KalimbaFlags.N.name: FlagRole.NegativeSignFlagRole,
        KalimbaFlags.V.name: FlagRole.OverflowFlagRole,
        KalimbaFlags.C.name: FlagRole.CarryFlagRole,
    }

    flag_write_types = [
        '*','zn','znvc'
    ]

    flags_written_by_flag_write_type = {
        '*' : [
            KalimbaFlags.Z.name,
            KalimbaFlags.N.name,
            KalimbaFlags.V.name,
            KalimbaFlags.C.name,
        ],
        'zn' : [
            KalimbaFlags.Z.name,
            KalimbaFlags.N.name,
        ],
        'znvc' : [
            KalimbaFlags.Z.name,
            KalimbaFlags.N.name,
            KalimbaFlags.V.name,
            KalimbaFlags.C.name,
        ],
	}

    flags_required_for_flag_condition = {
        LowLevelILFlagCondition.LLFC_E:   [KalimbaFlags.Z.name], 
        LowLevelILFlagCondition.LLFC_NE:  [KalimbaFlags.Z.name],
        LowLevelILFlagCondition.LLFC_NEG: [KalimbaFlags.N.name],
        LowLevelILFlagCondition.LLFC_POS: [KalimbaFlags.N.name],
        LowLevelILFlagCondition.LLFC_O:   [KalimbaFlags.V.name],
        LowLevelILFlagCondition.LLFC_NO:  [KalimbaFlags.V.name],
        LowLevelILFlagCondition.LLFC_UGT: [KalimbaFlags.C.name, KalimbaFlags.Z.name],
        LowLevelILFlagCondition.LLFC_ULE: [KalimbaFlags.C.name, KalimbaFlags.Z.name],
        LowLevelILFlagCondition.LLFC_SGE: [KalimbaFlags.N.name, KalimbaFlags.V.name],
        LowLevelILFlagCondition.LLFC_SLT: [KalimbaFlags.N.name, KalimbaFlags.V.name],
        LowLevelILFlagCondition.LLFC_SGT: [KalimbaFlags.Z.name, KalimbaFlags.N.name, KalimbaFlags.V.name],
        LowLevelILFlagCondition.LLFC_SLE: [KalimbaFlags.Z.name, KalimbaFlags.N.name, KalimbaFlags.V.name],
	}

    stack_pointer = 'SP'
    link_reg = 'rLink'
    minim_offset = 0#0x180
    
    def get_instruction_info(self, data:bytes, addr:int) -> Optional[InstructionInfo]:
        result = InstructionInfo()
        dec_len, dec_data = kalimba_minim_decode(data, addr)
        result.length = dec_len
        if not isinstance(dec_data, KalimbaControlFlow):
            return result
        
        if dec_data.cond == KalimbaCond.Always:
            if dec_data.op == KalimbaOp.RTS:
                result.add_branch(BranchType.FunctionReturn)
            elif dec_data.op == KalimbaOp.JUMP:
                if isinstance(dec_data.a, int):
                    result.add_branch(BranchType.UnconditionalBranch, (dec_data.a & -2) + addr)
                else:
                    result.add_branch(BranchType.IndirectBranch)
            elif dec_data.op == KalimbaOp.CALL:
                if isinstance(dec_data.a, int):
                    result.add_branch(BranchType.CallDestination, (dec_data.a & -2) + addr)
                else:
                    result.add_branch(BranchType.IndirectBranch)
        if dec_data.cond != KalimbaCond.Always:
            if dec_data.op == KalimbaOp.JUMP:
                if isinstance(dec_data.a, int):
                    result.add_branch(BranchType.TrueBranch, (dec_data.a & -2) + addr)
                    result.add_branch(BranchType.FalseBranch, addr + dec_len)
                else:
                    result.add_branch(BranchType.IndirectBranch)
            elif dec_data.op == KalimbaOp.CALL:
                if isinstance(dec_data.a, int):# no 'TrueCallDestination'
                    result.add_branch(BranchType.CallDestination, (dec_data.a & -2) + addr)
                    result.add_branch(BranchType.FalseBranch, addr + dec_len)
                else:
                    result.add_branch(BranchType.IndirectBranch)
            elif dec_data.op == KalimbaOp.DOLOOP:
                if isinstance(dec_data.a, int):
                    result.add_branch(BranchType.FalseBranch, (dec_data.a & -2) + addr)
                    result.add_branch(BranchType.TrueBranch, addr + dec_len)
                # TODO jump back
        return result

    def _padding(self, s=''):
        return InstructionTextToken(InstructionTextTokenType.TextToken, f'{s} ')

    def get_instruction_text(self, data: bytes, addr: int) -> Optional[Tuple[List[InstructionTextToken], int]]:
        dec_len, dec_data = kalimba_minim_decode(data, addr)
        ops = []
        if dec_data:
            ops.append(InstructionTextToken(InstructionTextTokenType.TextToken, dec_data.__str__()))
        return ops, dec_len

    def get_instruction_low_level_il(self, data: bytes, addr: int, il: LowLevelILFunction) -> Optional[int]:
        dec_len, dec_data = kalimba_minim_decode(data, addr)
        if dec_data:
            dec_data.llil(il, addr, dec_len)
        return dec_len
