import unittest
from unittest import TestCase
import clustering_eval_toolbox as tl
import pandas as pd
import numpy as np

class TestChisquare(TestCase):
    def setUp(self):
        self.phenotype_df_empty = pd.DataFrame(columns=['Cluster_ID', 'age'])
        self.phenotype_df =  pd.DataFrame(
            columns=['Cluster_ID', 'age'], index=['s'+str(i+1) for i in range(8)])
        self.phenotype_df['Cluster_ID'] = [0, 1, 3, 2, 1, 0, 1, 2]
        self.phenotype_df['age'] = [6, 18, 22, 6, 6, 22, 18, 18]

    def tearDown(self):
        del self.phenotype_df_empty
        del self.phenotype_df

    def test_chisquare(self):
        ret_empty = tl.chisquare(self.phenotype_df_empty)
        res_empty = ['chisquare', 0, 0, np.nan, np.nan]

        ret = tl.chisquare(self.phenotype_df)
        ret[3:] = ['%.6f' % e for e in ret[3:]]
        res = ['chisquare', 3, 8, '6.444444', '0.375285']
        
        self.assertEqual(np.array_equal(ret_empty, res_empty), True)
        self.assertEqual(np.array_equal(ret, res), True)


if __name__ == '__main__':
    unittest.main()