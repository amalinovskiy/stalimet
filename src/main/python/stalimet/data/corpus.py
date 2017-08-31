from stalimet.alignment.ibm2_exact import IBMModel2Exact
from nltk import AlignedSent


class SentencePair(AlignedSent):
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def __init__(self, words, mots, alignment=None):
        super(SentencePair, self).__init__(words, mots, alignment=alignment)
        self._score = 0.0

    def __str__(self):
        return str(self.score) + ' ||| ' + ' '.join(self.words) + ' ||| ' + ' '.join(self.mots)

    def align(self, model: IBMModel2Exact):
        model.align(self)


class ParallelCorpus(list):
    @property
    def alignment_model(self):
        return self._alignment_model

    def __init__(self):
        super(ParallelCorpus, self).__init__()
        self._alignment_model = None

    def build_parallel_corpus(self, tgt: list, ref: list) -> 'ParallelCorpus':
        for i in range(len(tgt)):
            self.append(SentencePair(tgt[i].split(), ref[i].split()))

        return self

    def train_alignment_model(self):
        self._alignment_model = IBMModel2Exact(self, 5)
        return self

