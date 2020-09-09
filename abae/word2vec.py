import gensim
import codecs


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main():
    source = '../sample_data/abae/train.txt'
    model_file = '../sample_data/abae/w2v_embedding'
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=50, workers=4)
    model.save(model_file)


print 'Pre-training word embeddings ...'
main()