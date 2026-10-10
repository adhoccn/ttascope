"""TTAScope 检测器：面向未知大语言模型的中文 AI 生成文本检测。

算法来自论文《面向未知大语言模型的中文 AI 生成文本检测：判别漂移的
可预测性与无标签批校正》，对应实验脚本 exp/E26 / E29 / E30。

    from detector.model import TTADetector
    det = TTADetector.load('models/ttascope_15d.joblib')
    det.detect(texts)
"""
__version__ = '1.0.0'

from .model import TTADetector          # noqa: F401
from . import tta, diagnostics, features  # noqa: F401
