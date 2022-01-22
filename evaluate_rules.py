import pandas as pd
import numpy as np
from tqdm import tqdm, trange

from snorkel.labeling import LabelingFunction
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.preprocess import preprocessor