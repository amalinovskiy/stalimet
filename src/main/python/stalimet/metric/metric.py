from stalimet.data.corpus import ParallelCorpus, SentencePair
from math import pow


class Metric(object):

    def __init__(self, tgt_ref_corpus: ParallelCorpus, ref_tgt_corpus: ParallelCorpus):
        self._tgt_ref_corpus = tgt_ref_corpus
        self._ref_tgt_corpus = ref_tgt_corpus
        self._alpha = 0.85
        self._beta = 0.2
        self._gamma = 0.6
        self._delta = 0.75

    @staticmethod
    def __count_chunks(sentence_pair: SentencePair) -> int:
        previous_index = -1
        chunks = 1
        for a in sorted(sentence_pair.alignment):
            if previous_index != -1 and (a[1] is None or previous_index != a[1] - 1):
                chunks += 1
            previous_index = a[1]

        return chunks

    @staticmethod
    def __count_matching_chars(word1: str, word2: str) -> int:
        count = 0
        for i, c in enumerate(word1):
            if len(word2) > i and c == word2[i]:
                count += 1

        return count

    @staticmethod
    def __calculate_char_match(sentence_pair: SentencePair) -> float:
        matches = 0.0
        for a in sorted(sentence_pair.alignment):
            count = 0 if a[1] is None else Metric.__count_matching_chars(sentence_pair.words[a[0]], sentence_pair.mots[a[1]])
            if count > 0:
                matches += count / len(sentence_pair.words[a[0]])
        return matches

    def __calculate_fragmentation_penaty(self, chunks, tgt_ref_match, ref_tgt_match):
        if chunks == 1:
            return 0

        pentalty = float(chunks) / ((tgt_ref_match + ref_tgt_match) / 2)
        pentalty = self._gamma * pow(pentalty, self._beta)
        return pentalty

    def calculate_scores(self):
        for i, sentence_pair in enumerate(self._tgt_ref_corpus):
            try:
                chunks = Metric.__count_chunks(sentence_pair)
                tgt_ref_match = Metric.__calculate_char_match(sentence_pair)
                ref_tgt_match = Metric.__calculate_char_match(self._tgt_ref_corpus[i])
                precision = tgt_ref_match / len(sentence_pair.words)
                recall = ref_tgt_match / len(sentence_pair.mots)
                fmean = 1.0 / (((1.0 - self._alpha) / precision) + (self._alpha / recall))
                penalty = self.__calculate_fragmentation_penaty(chunks, tgt_ref_match, ref_tgt_match)

                score = fmean * (1.0 - penalty)
            except ZeroDivisionError:
                score = 0

            sentence_pair.score = score
