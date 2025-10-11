import kalimba_bn

tests = [
    ('d84c', 0, 'rts'),
    ('da4c', 0, 'jump r0'),
    ('d24c', 0, 'call r0'),
    ('046e', 0x8080005c, 'if Always jump(m) 0x80800064'), # jump (m) $_s
    ('7501', 0, 'r3 = r4 + r3'),
    ('ea01', 0, 'r0 = r3 + r5'),
    ('00ea', 0, 'MB[Null + Null] = Null'),
    ('08ea', 0, 'MB[rMAC + Null] = Null'),
    ('10ea', 0, 'MB[r0 + Null] = Null'),
    ('18ea', 0, 'MB[r1 + Null] = Null'),
    ('20ea', 0, 'MB[r2 + Null] = Null'),
    ('28ea', 0, 'MB[r3 + Null] = Null'),
    ('30ea', 0, 'MB[r4 + Null] = Null'),
    ('38ea', 0, 'MB[r5 + Null] = Null'),
    ('80f000ea', 0, 'MB[r6 + Null] = Null'),
    ('80f000eb', 0, 'MB[r6 - Null] = Null'),
    ('90f000ea', 0, 'MB[r7 + Null] = Null'),
    ('a0f000ea', 0, 'MB[r8 + Null] = Null'),
    ('b0f000ea', 0, 'MB[r9 + Null] = Null'),
    ('c0f000ea', 0, 'MB[r10 + Null] = Null'),
    ('d0f000ea', 0, 'MB[rLink + Null] = Null'),
    ('f0f000ea', 0, 'MB[rMACB + Null] = Null'),
    ('e0f000ea', 0, 'MB[FP + Null] = Null'), # FP ones are tricky
    ('e0f400ea', 0, 'MB[L4 + Null] = Null'),
    ('e0ff00ea', 0, 'MB[L4 + Null] = I0'), # Same as below, bits are marked as 'x'
    ('e0fc00ea', 0, 'MB[L4 + Null] = I0'),
    ('14f003eb', 0, 'MB[rMAC - r1] = r2'),
    ('e4f003ea', 0, 'MB[FP + r1] = r2'),
    ('e9f0018a', 0, 'MB[FP + 0x1] = r7'),
    ('e27b', 0, 'M2 = -0x4'), # M2 = Null + -4;
    ('20f001f38ca2', 0, 'rMAC = MBU[Null + 0x41c8c]'),
    ('7ffcdcc8', 0x5f2, 'i32 r10 = r10 LSHIFT 0xfe'), # r10 = r10 LSHIFT -2
    ('ffff20ef', 0x8080008c, 'call 0x8080006c'), # call     $_maxim // 0x8080006c
    ('104e', 0x80800090, 'call(m) 0x808000b0'),  # call (m) $_minim // 0x808000b0
    ('106e', 0x80800090, 'if Always jump(m) 0x808000b0'),
    ('bfffffff2ced', 0x800074, 'if Always call 0x20'), # call 0x20
    ('bfffffff27ed', 0x80007a, 'if Always call(m) 0x20'), # call 0x21
    ('e1f002e8', 0, 'rMAC = M[FP + r0]'),
    ('0ff02a40', 0, 'rMACB = 0x2a'), # rMACB = Null + 42

    #('04f020f11ec9', 0, 'rMAC12 = rMAC ASHIFT 32'), # TODO: insert32
    ('04f020f11ec9', 0, 'rMAC12 = rMAC ASHIFT 32'), # rFlags = rMAC0
    #('3ff4c5c9', 0, 'r3 = r2 * r1 (int) (sat)'),
    #('3ff445c9', 0, 'r3 = r2 * r1 (frac)'),
    #('2ff6c1ca', 0, 'rMAC = rMAC + r4 * r0 (SS)'),
    #('30f050f2f6ca', 0, 'rMAC = rMAC + r4 * r0 (SS), r1 = M[I0,0], r3 = M[I4,0]'),
    #('80f001f811cf', 0, 'rMAC = M[SP + 0x1]'),
    #('03f811df', 0, 'rMAC = M[FP + 7]'),

    #('9ff242ce', 0, 'r0 = ONEBITCOUNT r0'), # TODO: maximode
    #('7ff242ce', 0, 'r0 = TWOBITCOUNT r0'),
    #('4ff242ce', 0, 'r0 = ABS r0'),
    #('5ff342ce', 0, 'r0 = MIN r1'),
    #('503062af', 0, 'rMAC = rMAC + r4 * r0 (SS), r1 = M[I0,0], r3 = M[I4,0]'),
]

def run_tests(testk):
    for (value, addr, expected) in tests:
        (tokens, length) = testk.get_instruction_text(bytes.fromhex(value), addr)
        actual = ' '.join([str(t) for t in tokens if str(t).strip()]).replace(' [ ', '[').replace(' ]', ']')
        assert actual == expected, f'Expected "{expected}", but got "{actual}"'

if __name__ == '__main__':
    testk = kalimba_bn.KALIMBA()
    run_tests(testk)
    #import pdb; pdb.set_trace()
    #print(testk.get_instruction_text(bytes.fromhex('0000'), 0))
