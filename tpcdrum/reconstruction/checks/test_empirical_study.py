"""ROOT-only integration of a frozen historical timing study, not detector accuracy."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reconstruction.calibration.study_empirical_timing import fit_exported
from reconstruction.stage_io import StageWriter, manifest, read_stage


def exported_fixture(directory, mismatch=None):
    """Only the two exported statistic ROOTs survive, with held-out offsets."""
    directory=Path(directory)
    original=directory/'original_digi.root'
    original.write_text('This original input is deliberately removed before fitting.')
    meta=dict(split_manifest_sha256='synthetic-frozen-manifest-before-data',
              observed_sources=[dict(path=str(original))],
              study_scope='synthetic software closure only',production_calibration_enabled=False)
    obs=directory/'observations.root';truth=directory/'truth.root'
    # Two training source groups, each with two siblings; disjoint held-out
    # groups have a deliberately nonzero offset that must not be fitted away.
    events=[(0,10),(0,10),(0,11),(0,11),(1,20),(1,21),(2,30),(2,31)]
    with StageWriter(obs,'empirical_timing_observations',dict(meta,role='observations')) as ow, \
            StageWriter(truth,'empirical_timing_truth',dict(meta,role='truth')) as tw:
        for entry,(split,group) in enumerate(events):
            if mismatch=='source_group_leak' and split==1:
                group=10+entry-4
            key=dict(file_index=np.int64(0),entry=np.int64(entry),source_group=np.int64(group),
                     split_code=np.int64(split))
            tb=100.+2.*entry+np.arange(12.)
            target=5.+.4*tb+{0:0.,1:7.,2:-3.}[split]
            observed_valid=np.ones(12,np.uint8);observed_valid[10]=0
            target_valid=np.ones(12,np.uint8);target_valid[11]=0
            # This outlier must be excluded by the OBSERVED gate; a NaN in the
            # unbracketed target layer must retain its row, not contaminate fit.
            target[10]=1e6;target[11]=np.nan
            ow.write(dict(key,mean_tb=tb,width_tb=np.full(12,4.),mean_u_mm=np.zeros(12),
                charge_adc_tb=np.full(12,1000.),edge_fraction_high=np.zeros(12),
                observed_layer_valid=observed_valid),dict(observed_only=True))
            target_key=dict(key)
            if mismatch=='paired_source_group' and entry==4:
                target_key['source_group']=np.int64(999)
            if mismatch=='paired_partition' and entry==4:
                target_key['split_code']=np.int64(2)
            point=np.column_stack([np.zeros(12),12.*np.arange(12),target])
            tw.write(dict(target_key,truth_crossing_uvw_mm=point,
                          truth_crossing_valid=target_valid),dict(evaluation_only=True))
    original.unlink()
    return obs,truth


class EmpiricalStudyROOTTests(unittest.TestCase):
    def test_train_only_fit_keeps_heldout_offsets_and_all_layer_denominators(self):
        import uproot
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);obs,truth=exported_fixture(directory)
            self.assertFalse((directory/'original_digi.root').exists())
            self.assertEqual(sorted(p.name for p in directory.iterdir()),['observations.root','truth.root'])
            result=fit_exported(obs,truth,directory/'fit',bootstrap_samples=10)
            model=json.loads((directory/'fit/frozen_affine_model.json').read_text())
            np.testing.assert_allclose(model['coefficients'],[5.,.4],atol=1e-11,rtol=0.)
            self.assertEqual(model['fit_source_groups'],[10,11])
            self.assertEqual(model['fit_row_count'],40)
            self.assertEqual(model['coverage']['total_rows'],48)
            self.assertEqual(model['coverage']['accepted_rows'],40)
            summaries=result['evaluations']['affine_train_only']
            for split,total,accepted,bias in [('train',48,40,0.),('validation',24,20,7.),('test',24,20,-3.)]:
                values=summaries[split]
                self.assertEqual(values['coverage']['total_rows'],total)
                self.assertEqual(values['coverage']['accepted_rows'],accepted)
                self.assertEqual(values['coverage']['rejected_rows'],total-accepted)
                self.assertAlmostEqual(values['summary']['bias_mm'],bias,places=10)
                self.assertFalse(values['coefficients_refitted'])
                if split!='train':
                    self.assertTrue(values['disjoint_from_training_source_groups'])
                    self.assertEqual(values['training_source_group_overlap_count'],0)
            self.assertEqual(result['coefficients_fitted_on'],'train only')
            self.assertFalse(result['calibrated_for_charge_mean'])
            self.assertFalse(result['eligible_for_production'])
            self.assertFalse(result['complete_objective_achieved'])
            closure=directory/'fit/timing_closure.root'
            self.assertEqual(manifest(closure)['role'],'evaluation')
            # Neither finite fitted coordinates nor the word "closure" makes
            # this evaluation ROOT an observation-only inference input.
            with self.assertRaisesRegex(ValueError,'observation-stage input'):
                list(read_stage(closure))
            rows=list(read_stage(closure,allow_truth=True));self.assertEqual(len(rows),8)
            for values,notes in rows:
                split=int(values['split_code'])
                self.assertEqual(values['observed_mean_tb'].shape,(12,))
                self.assertEqual(values['residual_true_minus_predicted_mm'].shape,(12,))
                np.testing.assert_allclose(values['residual_true_minus_predicted_mm'][:10],
                                           {0:0.,1:7.,2:-3.}[split],atol=1e-11,rtol=0.)
                self.assertFalse(values['observed_layer_valid'][10])
                self.assertFalse(values['target_layer_valid'][11])
                self.assertTrue(np.isnan(values['residual_true_minus_predicted_mm'][11]))
                self.assertFalse(notes['model_refitted'])
            with uproot.open(closure) as source:
                metrics=source['metrics'].arrays(library='np')
                counts=dict(zip(metrics['name'],metrics['value']))
                self.assertEqual(counts['possible_layers'],96.)
                self.assertEqual(counts['paired_layers'],80.)
                self.assertEqual(counts['production_calibration_enabled'],0.)
                for split,accepted in [('train',40),('validation',20),('test',20)]:
                    self.assertEqual(source['performance/affine_'+split+'_residual_mm'].values().sum(),accepted)

    def test_source_pair_or_partition_mismatch_and_group_leak_are_rejected(self):
        for mismatch,message in [('paired_source_group','pairing disagrees'),
                                 ('paired_partition','pairing disagrees'),
                                 ('source_group_leak','Source groups leaked')]:
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as folder:
                directory=Path(folder);obs,truth=exported_fixture(directory,mismatch=mismatch)
                with self.assertRaisesRegex(ValueError,message):
                    fit_exported(obs,truth,directory/'fit',bootstrap_samples=10)
                self.assertFalse((directory/'fit/frozen_affine_model.json').exists())
                self.assertFalse((directory/'fit/timing_closure.root').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
