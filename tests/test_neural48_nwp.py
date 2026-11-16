import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_neural48_nwp import sequence, bounded_mean, canonical, NATIVE


class ForecastSequenceTests(unittest.TestCase):
    def group(self):
        h=np.arange(6.,73.,3.)
        d=pd.DataFrame({'lead_hours':h})
        for name in NATIVE:d[name]=h
        d['precip_probability_6h']=np.where(h%6==0,.6,np.nan)
        d['qpf_category_6h']=np.where(h%6==0,(h/6)%7,np.nan)
        return d

    def test_hourly_mean_of_native_linear_forecast(self):
        x=sequence(self.group(),11.)
        np.testing.assert_allclose(x[:,0],np.arange(11.5,59.5))
        self.assertEqual(x.shape,(48,16))

    def test_categorical_qpf_preserved_across_half_hour(self):
        x=sequence(self.group(),12.5)
        # Interval17:30–18:30 splits the18h and24h6hcategories equally.
        self.assertAlmostEqual(float(x[5,6]),3.5)
        self.assertAlmostEqual(float(x[5,10]),.5)
        self.assertAlmostEqual(float(x[5,11]),.5)
        self.assertAlmostEqual(float(x[5,7:14].sum()),1.)
        self.assertAlmostEqual(float(x[5,5]),.6,places=6)

    def test_no_missing_interpolation_or_extrapolation(self):
        d=self.group();d.loc[d.lead_hours.eq(15),'temperature_c']=np.nan
        x=sequence(d,11.)
        self.assertTrue(np.isnan(x[1:7,0]).all())
        self.assertTrue(np.isfinite(x[7,0]))
        x=sequence(self.group(),25.)
        self.assertTrue(np.isnan(x[-1,:14]).all())

    def test_issue_before_target(self):
        with self.assertRaises(ValueError):sequence(self.group(),9.)

    def test_mixed_source_timestamp_strings_preserve_utc(self):
        d=pd.DataFrame({'runtime':['2026-11-09 00:00:00+00:00','2026-11-10 00:00:00'],
            'ftime':['2026-11-09 06:00:00+00:00','2026-11-10 06:00:00'],
            'tmp':[32,999],'dpt':[20,999],'wdr':[180,990],'wsp':[10,99],
            'p06':[60,999],'q06':[2,9]})
        x=canonical(d)
        self.assertTrue(str(x.runtime.dtype).endswith('UTC]'))
        self.assertEqual(x.temperature_c.iloc[0],0.)
        self.assertTrue(np.isnan(x.temperature_c.iloc[1]))
        self.assertTrue(np.isnan(x.qpf_category_6h.iloc[1]))
        self.assertEqual(x.lead_hours.to_list(),[6.,6.])


if __name__=='__main__':unittest.main()
