#
# Copyright 2016-2017, Q.I. Leap Analytics Inc., all rights reserved.
#

"""
Transformer base class.
**************************

**Authors:**
    charles@qileap

**File:** `transform_base.py`
"""

from abc import ABC, abstractmethod


class TransformerBase(ABC):
    @abstractmethod
    def transform_all(self, iterable):
        pass

    @abstractmethod
    def transform_one(self, single):
        pass

    def register_row_encoder(self, encoder):
        pass

    def register_column_encoder(self, encoder):
        pass

    @abstractmethod
    def configure(self, **kwargs):
        pass
