from binaryninja.callingconvention import CallingConvention
from .k_arch import KALIMBA

KALIMBA.register()

from .k_view import KALIMBAView
KALIMBAView.register()

class DefaultCallingConvention(CallingConvention):
    name = 'default'
    int_arg_regs = ['r0', 'r1', 'r2', 'r3']
    int_return_reg = 'r0'
    high_int_return_reg = 'r0'


#arch = Architecture[KALIMBA.name]
#arch.register_calling_convention(DefaultCallingConvention(arch, 'default'))
#arch.standalone_platform.default_calling_convention = arch.calling_conventions['default']

