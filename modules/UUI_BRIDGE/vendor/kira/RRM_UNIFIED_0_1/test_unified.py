import os, tempfile, unittest
from unified import UnifiedRuntime

def state(relation='near', a='a', b='b'):
    return {'nodes':{a:'point',b:'point'},'edges':[[a,relation,b]]}

class UnifiedTests(unittest.TestCase):
    def test_binary_operations_across_cache_sizes(self):
        for budget in (0, 32, 4096):
            with self.subTest(budget=budget):
                rt=UnifiedRuntime(cache_budget=budget)
                a=rt.store_state(state()); b=rt.store_state(state('far'))
                delta=rt.ops.diff(a,b)
                self.assertEqual(len(delta),1)
                new=rt.store_state(state(a='x',b='y'))
                result=rt.ops.continue_with(new,delta[0])
                self.assertEqual(rt.load_state(result),state('far','x','y'))
                self.assertEqual(rt.stats()['observations'],0)
                self.assertLessEqual(rt.cache_stats()['resident_wire_bytes'],budget)
                rt.close()

    def test_persistent_learning_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'kira.sqlite')
            rt=UnifiedRuntime(path,cache_budget=256)
            for _ in range(3):
                rt.observe('s',state('near'))
                rt.observe('s',state('far'))
            # Evidence semantics are inherited unchanged; general mode can predict.
            prediction=rt.predict('s',state('near'),contextual=False)
            self.assertEqual(prediction['status'],'HYPOTHESIS')
            self.assertEqual(prediction['result'],state('far'))
            before=rt.stats()
            rt.close()
            rt=UnifiedRuntime(path,cache_budget=0)
            self.assertEqual(rt.stats(),before)
            self.assertEqual(rt.predict('s',state('near'),contextual=False)['result'],state('far'))
            rt.close()

    def test_cache_hits_and_boundedness(self):
        rt=UnifiedRuntime(cache_budget=1024)
        handle=rt.store_state(state())
        rt._get(handle); rt._get(handle)
        self.assertGreater(rt.cache_stats()['hits'],0)
        self.assertLessEqual(rt.cache_stats()['resident_wire_bytes'],1024)
        rt.close()

    def test_dream_not_observation(self):
        rt=UnifiedRuntime()
        for _ in range(3):
            rt.observe('s',state()); rt.observe('s',state('far'))
        prior=rt.stats()
        rt.dream('s',state(),steps=3,contextual=False)
        self.assertEqual(rt.stats(),prior)
        rt.close()

if __name__=='__main__': unittest.main()
