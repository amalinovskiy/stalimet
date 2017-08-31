from stalimet.data.corpus import ParallelCorpus
from stalimet.preprocess.preprocess import run_preprocess
from stalimet.metric.metric import Metric

def main():
    tgt = run_preprocess('../../../../sample_data/tgt.txt', 'english')
    ref = run_preprocess('../../../../sample_data/ref.txt', 'english')
    parallel_corpus1 = ParallelCorpus()
    parallel_corpus1.build_parallel_corpus(tgt, ref)
    parallel_corpus1.train_alignment_model()

    parallel_corpus2 = ParallelCorpus()
    parallel_corpus2.build_parallel_corpus(ref, tgt)
    parallel_corpus2.train_alignment_model()

    metric = Metric(parallel_corpus1, parallel_corpus2)
    metric.calculate_scores()

    for entry in parallel_corpus1:
        print('{} {}'.format(str(entry), " ".join("%s-%s" % (str(p[0]), str(p[1])) for p in sorted(entry.alignment))))


if __name__ == '__main__':
    main()

