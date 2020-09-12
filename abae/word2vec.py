import gensim
import codecs
import argparse


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(text_type):
    source = '../sample_data/abae/'+text_type+'/train.txt'
    model_file = '../sample_data/abae/'+text_type+'/w2v_embedding'
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=50, workers=4)
    model.save(model_file)


parser = argparse.ArgumentParser()
parser.add_argument("--texttype", dest="text_type", type=str, metavar='<str>', default='all')
args = parser.parse_args()

print 'Pre-training word embeddings ...'
main(args.text_type)