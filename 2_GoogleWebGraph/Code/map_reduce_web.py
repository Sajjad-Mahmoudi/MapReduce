from mrjob.job import MRJob
from mrjob.step import MRStep
import re

pattern = re.compile(r"^(?!#).*")


class reverse_webLink(MRJob):

    # mapper creates a list of source_id and target_id web pages for each line
    # and returns the reversed list
    def mapper_get_pairs(self, _, line):
        pair = pattern.findall(line)
        if pair:
            pair = pair[0].split('\t')
            source = pair[0]
            target = pair[1]
            yield target, source

    # reducer returns a list of sources for each target
    def reducer_count_words(self, target, source):
        yield target, list(source)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_pairs),
            MRStep(reducer=self.reducer_count_words)
        ]


if __name__ == '__main__':
    reverse_webLink.run()
