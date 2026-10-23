import unittest
from uui_cognition import WILLI_RELATION_ALIASES, RELATION_OPS
from uui_lang import tokenize

class WilliAliasesTest(unittest.TestCase):
    def test_all_aliases_registered(self):
        self.assertEqual(len(WILLI_RELATION_ALIASES), 7)
        self.assertTrue(set(WILLI_RELATION_ALIASES).issubset(RELATION_OPS))
    def test_aliases_lex_as_identifiers(self):
        for alias in WILLI_RELATION_ALIASES:
            self.assertEqual(tokenize(alias)[0].value, alias)
