from mrjob.job import MRJob
from mrjob.step import MRStep
from math import sqrt


class FrobeniusNorm(MRJob):

    # the mapper produces 50 pairs for each row of the matrix as (row index, number^2)
    def mapper_get_pairs(self, _, line):
        row = line.split()
        for i in range(1, len(row)):
            yield int(row[0]), float(row[i])**2

    # this reducer produces sum of squared numbers of each row
    def reducer_sum_row(self, row_index, squaredNum):
        yield None, sum(squaredNum)

    # this reducer yields the Frobenius norm
    def reducer_Fnorm(self, _, squaredSum):
        yield 'Frobenius Norm', sqrt(sum(squaredSum))

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_pairs,
                   reducer=self.reducer_sum_row),
            MRStep(reducer=self.reducer_Fnorm)
        ]


if __name__ == "__main__":
    FrobeniusNorm.run()