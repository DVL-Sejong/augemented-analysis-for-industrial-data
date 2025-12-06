import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def calculate_reconstruction_error_statistics(recon_errors: np.ndarray) -> Dict[str, float]:
