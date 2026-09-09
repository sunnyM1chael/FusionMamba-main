import tempfile
from pathlib import Path
import unittest

from split_manifest import read_manifest, paired_paths
from evaluate_fusion import evaluation_samples


class ManifestIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root/'test.txt'
        for name in ('ir','vis','fused'):
            (self.root/name).mkdir()

    def test_weighting_ablation_modes_are_complementary(self):
        import torch
        try:
            from models.cross import AlignmentConfidenceGuidedAdaptiveWeighting
        except ModuleNotFoundError as exc:
            self.skipTest(f'optional model dependency unavailable: {exc.name}')
        torch.manual_seed(1)
        infrared, visible = torch.randn(2,16,5,7), torch.randn(2,16,5,7)
        for mode in ('equal','learned','acgaw'):
            fused, weights = AlignmentConfidenceGuidedAdaptiveWeighting(16,mode=mode)(infrared,visible)
            self.assertEqual(fused.shape,infrared.shape)
            self.assertTrue(torch.allclose(weights.sum(1),torch.ones_like(infrared),atol=1e-6))
        with self.assertRaises(ValueError):
            AlignmentConfidenceGuidedAdaptiveWeighting(16,mode='unknown')

    def test_reject_duplicate_and_traversal(self):
        for text in ('a.png\na.png\n', '../a.png\n', '/a.png\n', 'C:/a.png\n', '# empty\n'):
            self.manifest.write_text(text)
            with self.assertRaises(ValueError):
                read_manifest(self.manifest)

    def test_missing_pair_is_not_silently_dropped(self):
        (self.root/'ir'/'a.png').touch()
        with self.assertRaises(ValueError):
            paired_paths(self.root/'ir', self.root/'vis', ['a.png'])

    def test_strict_nested_evaluation(self):
        self.manifest.write_text('test/a.png\n')
        for folder in ('ir','vis','fused'):
            (self.root/folder/'test').mkdir()
            (self.root/folder/'test'/'a.png').touch()
        args = [self.root/n for n in ('ir','vis','fused')]
        self.assertEqual(len(evaluation_samples(*args,self.manifest)),1)
        (self.root/'fused'/'extra.png').touch()
        with self.assertRaises(ValueError):
            evaluation_samples(*args,self.manifest)

    def test_missing_fused_output_fails_without_manifest(self):
        for folder in ('ir','vis'):
            (self.root/folder/'a.png').touch()
        with self.assertRaises(ValueError):
            evaluation_samples(*[self.root/n for n in ('ir','vis','fused')])

    def test_loader_does_not_drop_missing_flat_name(self):
        from TaskFusion_dataset import Fusion_dataset
        for folder in ('ir','vis'):
            (self.root/folder/'a.png').touch()
        self.manifest.write_text('a.png\nmissing.png\n')
        with self.assertRaises(ValueError):
            Fusion_dataset('train', str(self.root/'ir'), str(self.root/'vis'),
                           split_file=str(self.manifest))

    def test_nested_names_survive_generation_and_loader(self):
        from TaskFusion_dataset import Fusion_dataset
        from generate_fused_dataset import selected_names
        self.manifest.write_text('train/a.png\n')
        for folder in ('ir','vis'):
            (self.root/folder/'train').mkdir()
            (self.root/folder/'train'/'a.png').touch()
        a,b,names = selected_names(self.root/'ir',self.root/'vis',self.manifest)
        self.assertEqual(names,['train/a.png'])
        dataset = Fusion_dataset('train',str(self.root/'ir'),str(self.root/'vis'),split_file=self.manifest)
        self.assertEqual(len(dataset),1)
        self.assertEqual(dataset.filenames_ir,names)


if __name__ == '__main__':
    unittest.main()
