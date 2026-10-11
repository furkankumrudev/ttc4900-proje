"""Faz 0 duman testi: paket src/ altından import edilebiliyor mu?"""

import ttc4900


def test_package_imports():
    assert ttc4900.__version__
