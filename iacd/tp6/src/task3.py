import sys
import shlex
sys.modules['pipes'] = shlex
import re

from mrjob.job import MRJob
from mrjob.step import MRStep

# Expressão regular para extrair apenas palavras
WORD_REGEXP = re.compile(r"[\w']+")

class MRWordFrequencySorted(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper=self.mapper_make_counts_key,
                   reducer=self.reducer_output_words)
        ]

    def mapper_get_words(self, _, line):
        words = WORD_REGEXP.findall(line)
        for word in words:
            yield word.lower(), 1

    def reducer_count_words(self, word, values):
        yield word, sum(values)

    def mapper_make_counts_key(self, word, count):
        # Formata o número com 4 zeros (ex: 0230) para o Hadoop ordenar corretamente por ordem crescente
        yield '%04d' % int(count), word

    def reducer_output_words(self, count, words):
        for word in words:
            yield word, int(count)

if __name__ == '__main__':
    MRWordFrequencySorted.run()