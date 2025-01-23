from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ScalarArray = FloatArray | IntArray
FloatOrArray = TypeVar("FloatOrArray", float, FloatArray)
