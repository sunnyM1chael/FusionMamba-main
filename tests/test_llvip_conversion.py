import tempfile
from pathlib import Path
import unittest

from research.prepare_llvip_yolo import convert_annotation


class LlvipConversionTest(unittest.TestCase):
    def test_valid_and_zero_width_box(self):
        xml = '''<annotation><size><width>100</width><height>50</height></size>
        <object><name>person</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox></object>
        <object><name>person</name><bndbox><xmin>40</xmin><ymin>5</ymin><xmax>40</xmax><ymax>6</ymax></bndbox></object></annotation>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'a.xml'
            path.write_text(xml)
            rows,bad = convert_annotation(path)
        self.assertEqual(rows,['0 0.20000000 0.30000000 0.20000000 0.40000000'])
        self.assertEqual(len(bad),1)
        self.assertEqual(bad[0]['reason'],'non_positive_extent')


if __name__ == '__main__':
    unittest.main()
