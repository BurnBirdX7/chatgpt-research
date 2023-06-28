"""
Searcher is moved to a separate module, so it won't be loaded with other parts
"""

import lucene
from .Searcher import Searcher
lucene.initVM(vmargs=['-Djava.awt.headless=true'])
