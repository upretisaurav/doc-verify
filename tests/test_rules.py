"""The generator's consistency rules: clean documents pass every rule, altered ones break exactly their intended rule."""

import pytest

from sad_syn.doc_types import DOC_TYPES, rule_test
from sad_syn.generate import set_seed

N = 200


@pytest.mark.parametrize('doc_type', list(DOC_TYPES))
def test_every_document_breaks_exactly_its_intended_rule(doc_type):
    set_seed(0)
    counts = rule_test(doc_type, N)
    assert counts == {kind: N for kind in counts}
