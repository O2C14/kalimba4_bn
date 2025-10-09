import kalimba_bn

if __name__ == '__main__':
    testk = kalimba_bn.KALIMBA()
    #print(testk.get_instruction_text(bytes.fromhex('e27b'), 0))
    #print(testk.get_instruction_text(bytes.fromhex('20f001f38ca2'),0))
    print(testk.get_instruction_text(bytes.fromhex('7ffcdcc8'), 0x5f2))
    print(testk.get_instruction_text(bytes.fromhex('c4 4e'), 0x8000107c))#80001204
    print(testk.get_instruction_text(bytes.fromhex('ee 4f'), 0x800003d2))#800003ae