"""特征集。与论文 §3.2 的三套表征一一对应，接口统一为 fit / transform。

    15D        15 维手工特征，零外部依赖
    char_2_4   字符 2-4gram TF-IDF，经 TruncatedSVD 降到 SVD_DIM 维
    roberta    冻结 chinese-roberta-wwm-ext 的 768 维 mean-pooling（需 torch）

TF-IDF 一律 fit-on-train-only：词表、IDF、SVD 只由训练语料估计，
测试批次只做 transform。
"""
import numpy as np

SVD_DIM = 50

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
except ImportError:                                     # pragma: no cover
    TfidfVectorizer = TruncatedSVD = None

from .feature_extractor_15d import extract_features_15d


class Feat15D:
    name = '15D'

    def fit(self, texts):
        return self

    def transform(self, texts):
        out = []
        for t in texts:
            try:
                out.append(extract_features_15d(t))
            except Exception:
                out.append(np.full(15, np.nan))
        return np.asarray(out, dtype=np.float64)


class CharTfidf:
    def __init__(self, analyzer='char', ngram=(2, 4), max_features=30000, min_df=5):
        self.analyzer, self.ngram = analyzer, ngram
        self.max_features, self.min_df = max_features, min_df
        self.name = '%s_%d_%d' % (analyzer, ngram[0], ngram[1])

    def fit(self, texts):
        if self.analyzer == 'word':
            self.vec = TfidfVectorizer(max_features=self.max_features,
                                       ngram_range=self.ngram,
                                       min_df=self.min_df, max_df=0.95)
            corpus = list(texts)
        else:
            self.vec = TfidfVectorizer(analyzer='char', ngram_range=self.ngram,
                                       max_features=self.max_features,
                                       min_df=self.min_df, sublinear_tf=True)
            corpus = list(texts)
        A = self.vec.fit_transform(corpus)
        self.svd = (TruncatedSVD(n_components=SVD_DIM, random_state=42).fit(A)
                    if A.shape[1] > SVD_DIM else None)
        return self

    def transform(self, texts):
        A = self.vec.transform(list(texts))
        if self.svd is None:
            return np.asarray(A, dtype=np.float64)
        return np.asarray(self.svd.transform(A), dtype=np.float64)


class RobertaFrozen:
    """冻结编码器取 embedding，不做微调。无 torch 时 build() 会自动跳过。"""

    name = 'roberta_frozen'

    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', batch_size=16):
        self.model_name, self.batch_size = model_name, batch_size

    def fit(self, texts):
        return self

    def transform(self, texts):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not hasattr(self, '_tok'):
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = [str(t)[:512] for t in texts[i:i + self.batch_size]]
                enc = self._tok(chunk, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
                h = self._model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                outs.append(pooled.numpy())
        return np.vstack(outs).astype(np.float64)


FEATURES = {
    '15D': lambda: Feat15D(),
    'char_2_4': lambda: CharTfidf('char', (2, 4), 30000, 5),
    'roberta_frozen': lambda: RobertaFrozen(),
}


def build(name):
    if name not in FEATURES:
        raise KeyError('未知特征集 %s，可选：%s' % (name, ', '.join(FEATURES)))
    return FEATURES[name]()
