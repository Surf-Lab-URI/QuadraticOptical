"""CLI integration, comparison immutability, reporting and failure contracts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

from quadratic_optical import cli, reporting
from quadratic_optical.matio import read_native_piv


def invoke(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(args)
    return code, out.getvalue(), err.getvalue()


class CLIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='qo_cli_')
        cls.base = Path(cls.temporary.name)
        code, out, err = invoke(['demo', '--output', str(cls.base/'demo'), '--workers', '2'])
        if code:
            cls.temporary.cleanup()
            raise AssertionError('Full synthetic CLI demo failed:\n'+out+'\n'+err)
        cls.demo = cls.base/'demo'
        cls.directory = cls.demo/'results'/'Demo_0'
        cls.piv = cls.demo/'input'/'Demo_0_PIV.mat'
        cls.initial_summary=json.loads((cls.demo/'results'/'batch_summary.json').read_text())
        cls.initial_html=(cls.directory/'index.html').read_text()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_demo_discovery_schema_and_report(self):
        code, out, err = invoke(['discover', str(self.demo/'input')])
        self.assertEqual(code, 0, err)
        discovered = json.loads(out)
        self.assertEqual(discovered['pairs'][0]['name'], 'Demo_0')
        native = read_native_piv(self.piv)
        self.assertEqual(native['dx_px'].shape, (39, 47))
        np.testing.assert_array_equal(native['disp_px'][0, 0], [6, -3])
        summary = self.initial_summary
        self.assertEqual(summary['pairs'][0]['status'], 'complete')
        self.assertEqual(summary['pairs'][0]['prediction_status'], 'complete')
        self.assertFalse((self.directory/'.processing.lock').exists())
        for name in ['index.html','quiver.png','quiver.svg','profiles.png','gradient_comparison.png',
                     'velocity_gradients.mat','velocity_gradients.csv','piv_comparison.mat']:
            self.assertGreater((self.directory/name).stat().st_size, 0)
        export=loadmat(self.directory/'velocity_gradients.mat',simplify_cells=True)
        n=len(export['u_m_per_s'])
        self.assertEqual(export['gradient_px_per_px'].shape,(n,2,2))
        self.assertLess(abs(np.nanmedian(export['u_m_per_s'])-.06),1e-4)
        self.assertLess(abs(np.nanmedian(export['w_m_per_s'])-.03),1e-4)
        self.assertIn('finite correlation required',self.initial_html)
        metadata=json.loads((self.directory/'quiver_display_metadata.json').read_text())
        with np.load(self.directory/'results.npz') as archive:
            displacement=archive['disp'];dx=float(archive['DX']);accepted=archive['accepted']
        for series in metadata['series']:
            origins=np.asarray(series['origin_cm']);tips=np.asarray(series['displayed_tip_cm'])
            if not len(origins):continue
            for points in [origins,tips]:
                self.assertTrue(np.all(points[:,0]>metadata['xlim_cm'][0]))
                self.assertTrue(np.all(points[:,0]<metadata['xlim_cm'][1]))
                self.assertTrue(np.all(points[:,1]>metadata['ylim_cm'][0]))
                self.assertTrue(np.all(points[:,1]<metadata['ylim_cm'][1]))
            if series['method']=='Image-only optical flow':
                indices=np.asarray(series['source_indices'],int)
                self.assertTrue(accepted[indices].all())
                expected=displacement[indices]*np.array([1.,-1.])*dx*100*metadata['arrow_gain']
                np.testing.assert_allclose(tips-origins,expected,atol=1e-13)

    def test_resume_changes_workers_and_quality_without_refitting(self):
        from quadratic_optical.core import tracking, fit_fields
        with patch.object(tracking.ImageOnlyTracker,'coarse_task',side_effect=AssertionError('Unexpected coarse refit')), \
             patch.object(fit_fields,'fit_local',side_effect=AssertionError('Unexpected local refit')):
            code, out, err=invoke(['run',str(self.demo/'input'),'--output',str(self.demo/'results'),
                                  '--config',str(self.demo/'demo.json'),'--workers','1','--piv-quality','finite'])
        self.assertEqual(code,0,out+err)
        self.assertIn('supplied finite vectors',(self.directory/'index.html').read_text())

    def test_compare_ir_is_persistent_strict_and_does_not_change_fields(self):
        immutable=['inputs','results','plot_samples','integration_profile','main','affine','large_window','margin14','reverse','ptv_tracks']
        before={n:hashlib.sha256((self.directory/(n+'.npz')).read_bytes()).hexdigest() for n in immutable}
        ir=self.base/'ir_reference.mat'
        savemat(ir,{'exp_name':'Demo','PIV':{'pairNum':[0],'IR_idx':[2],'t':[.01]},
                    'USurf':{'t':[0.,.01,.02],'usurf0':[.02,np.nan,.03],
                             'usurffilt':[.021,.025,.031]}})
        code,out,err=invoke(['compare',str(self.directory),'--piv',str(self.piv),
                             '--ir-results',str(ir),'--experiment','Demo','--pair-number','0'])
        self.assertEqual(code,0,out+err)
        record=json.loads((self.directory/'surface_ir_reference.json').read_text(),
                          parse_constant=lambda s: self.fail('Nonstandard JSON constant '+s))
        self.assertEqual(record['selected_field'],'USurf.usurffilt')
        self.assertEqual(record['velocity_m_per_s'],.025)
        self.assertEqual(record['IR_index_matlab'],2)
        self.assertTrue(record['comparison_only'])
        code,out,err=invoke(['compare',str(self.directory),'--piv',str(self.piv),'--quality','finite'])
        self.assertEqual(code,0,out+err)
        self.assertEqual(record,json.loads((self.directory/'surface_ir_reference.json').read_text()))
        after={n:hashlib.sha256((self.directory/(n+'.npz')).read_bytes()).hexdigest() for n in immutable}
        self.assertEqual(before,after)

    def test_optional_comparison_failure_preserves_prediction_report(self):
        with patch('quadratic_optical.comparison.compare_pair',side_effect=ValueError('Synthetic comparison failure')):
            code,out,err=invoke(['run',str(self.demo/'input'),'--output',str(self.demo/'results'),
                                 '--config',str(self.demo/'demo.json')])
        self.assertEqual(code,1)
        row=json.loads((self.directory/'status.json').read_text())
        self.assertEqual(row['stage'],'comparison')
        self.assertEqual(row['prediction_status'],'complete')
        self.assertEqual(row['status'],'failed')
        self.assertTrue(row['prediction_report_available'])
        self.assertTrue((self.directory/'index.html').is_file())
        self.assertFalse((self.directory/'.processing.lock').exists())
        # An independent comparison retry also repairs the failed comparison
        # status without invoking the estimator again.
        code,out,err=invoke(['compare',str(self.directory),'--piv',str(self.piv)])
        self.assertEqual(code,0,out+err)
        self.assertEqual(json.loads((self.directory/'status.json').read_text())['status'],'complete')

    def test_conflicting_inputs_and_existing_lock_fail_without_overwrite(self):
        config=json.loads((self.demo/'demo.json').read_text());config['depth_m']=.005
        changed=self.base/'changed.json';changed.write_text(json.dumps(config))
        before=hashlib.sha256((self.directory/'inputs.npz').read_bytes()).hexdigest()
        code,out,err=invoke(['run',str(self.demo/'input'),'--output',str(self.demo/'results'),
                             '--config',str(changed)])
        self.assertEqual(code,1)
        self.assertIn('different inputs/settings',err)
        self.assertEqual(before,hashlib.sha256((self.directory/'inputs.npz').read_bytes()).hexdigest())
        lock=self.directory/'.processing.lock';lock.write_text('{"pid": 123}')
        try:
            code,out,err=invoke(['compare',str(self.directory),'--piv',str(self.piv)])
            self.assertEqual(code,1)
            self.assertIn('locked',err)
            self.assertEqual(lock.read_text(),'{"pid": 123}')
        finally:
            lock.unlink()


class CLIUtilityTests(unittest.TestCase):
    def test_continue_on_error_and_previous_batch_entries_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);inputs=base/'input';inputs.mkdir();output=base/'output';output.mkdir()
            for pair in ['Test_0','Test_1']:
                for frame in 'AB':(inputs/(pair+'_img'+frame+'.tif')).write_bytes(b'fixture')
            (output/'batch_summary.json').write_text(json.dumps({'pairs':[{'pair':'Earlier','status':'complete'}]}))
            with patch('quadratic_optical.prepare.prepare',side_effect=ValueError('Deliberate invalid image fixture')) as prepare:
                code,out,err=invoke(['run',str(inputs),'--output',str(output),'--continue-on-error'])
            self.assertEqual(code,1)
            self.assertEqual(prepare.call_count,2)
            rows=json.loads((output/'batch_summary.json').read_text())['pairs']
            self.assertEqual([r['pair'] for r in rows],['Earlier','Test_0','Test_1'])
            self.assertEqual([r['status'] for r in rows],['complete','failed','failed'])
            self.assertFalse(any(output.glob('*/.processing.lock')))

    def test_demo_generate_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            code,out,err=invoke(['demo','--output',td,'--generate-only'])
            self.assertEqual(code,0,out+err)
            code,out,err=invoke(['demo','--output',td,'--generate-only'])
            self.assertEqual(code,1)
            self.assertIn('already exist',err)

    def test_strict_surface_annotation_rejects_theory_and_nonfinite_velocity(self):
        with tempfile.TemporaryDirectory() as td:
            for record in [dict(available=True,selected_field='USurf.msv98',velocity_m_per_s=.1,comparison_only=True),
                           dict(available=True,selected_field='USurf.usurf0',velocity_m_per_s=np.nan,comparison_only=True)]:
                with self.assertRaises(ValueError):reporting.surface_reference(td,record)
            reporting.write_json(Path(td)/'strict.json',dict(missing=np.nan,valid=np.float64(.2)))
            self.assertEqual(json.loads((Path(td)/'strict.json').read_text()),dict(missing=None,valid=.2))


if __name__=='__main__':unittest.main()
