import unittest

from research.prepare_protocol import split_llvip, split_msrs


class ProtocolSplitTest(unittest.TestCase):
    def test_msrs_deterministic_disjoint_and_illumination(self):
        names = [f'{i:05d}{light}.png' for light in ('D','N') for i in range(100)]
        a,b = split_msrs(names,42)
        self.assertEqual((a,b),split_msrs(list(reversed(names)),42))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a)|set(b),set(names))
        self.assertEqual(sum(n.endswith('D.png') for n in b),10)
        self.assertEqual(sum(n.endswith('N.png') for n in b),10)

    def test_llvip_preserves_complete_groups(self):
        names = [f'{g:02d}{i:04d}.jpg' for g in range(1,11) for i in range(50)]
        a,b,groups = split_llvip(names,42)
        self.assertEqual((a,b,groups),split_llvip(list(reversed(names)),42))
        self.assertFalse({n[:2] for n in a}&{n[:2] for n in b})
        self.assertEqual(set(a)|set(b),set(names))
        self.assertEqual(len(b),50)


if __name__ == '__main__':
    unittest.main()
