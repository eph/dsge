import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase



import dill
dill.settings['recurse'] = True
#dill.detect.trace(True)

import pkg_resources

from dsge.examples import nkmp as dsge1
p0 = dsge1.p0()

nkmp = dsge1.compile_model()


# class TestPickle(TestCase):

#     def test_abcd(self):
#         dill.dumps(nkmp.log_lik)
#         self.assertEqual(0,0)


#     def test_parallel(self):
#         from pathos.multiprocessing import ProcessPool as Pool
#         dill.settings['recurse'] = True        
#         p = Pool(4)
#         l = lambda x: nkmp.log_lik(x)
#         r = p.map(nkmp.prior.logpdf, np.array(4*[p0]))
#         r = p.map(l, np.array(1000*[p0]))

        
