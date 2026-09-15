# pycraft/__init__.py
#
# PyCraft-1: a 55M-parameter Python code LLM trained from scratch.
#
#     from pycraft import PyCraft
#     pc = PyCraft()
#     print(pc.generate("def is_palindrome(s):"))

from pycraft.engine import PyCraft, strip_fences

__version__ = "0.1.0"
__all__ = ["PyCraft", "strip_fences"]
