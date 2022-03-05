from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import heapq
import pandas as pd
from math import sqrt

pattern = re.compile(r'^[0-9].*[a-z]$')  # using this pattern, we can skip the header row and the unknown samples

df = pd.read_csv('KNN/Iris.csv')  # convert .csv file to dataframe in order to get the extremum of each feature
mini = df.iloc[:, 1:5].agg(['min', 'max']).values.tolist()[0]  # list of minimums of features
maxi = df.iloc[:, 1:5].agg(['min', 'max']).values.tolist()[1]  # list of maximums of features

list_query_id = [39, 54, 96]  # list of IDs of unknown samples/queries

list_query_withSpecies = df.loc[df['Id'].isin(list_query_id)].values.tolist()
query_39 = list_query_withSpecies[0][:-1]  # list of ID + features of query 39
query_54 = list_query_withSpecies[1][:-1]  # list of ID + features of query 54
query_96 = list_query_withSpecies[2][:-1]  # list of ID + features of query 96

# list of ID + normalized features of query 39
query_39_normalized = [query_39[0]] + [(query_39[i+1] - mini[i]) / (maxi[i] - mini[i]) for i in range(4)]
# list of ID + normalized features of query 54
query_54_normalized = [query_54[0]] + [(query_54[i+1] - mini[i]) / (maxi[i] - mini[i]) for i in range(4)]
# list of ID + normalized features of query 96
query_96_normalized = [query_96[0]] + [(query_96[i+1] - mini[i]) / (maxi[i] - mini[i]) for i in range(4)]


class KNN(MRJob):

    # the mapper produces 4 pairs for each sample as
    # ((ID, label), squared difference between each feature of all samples and corresponding query feature)
    def mapper_get_pairs(self, _, line):
        pair = pattern.findall(line)
        if pair:
            pair = pair[0].split(',')
            for i in range(1, 5):
                yield (float(pair[0]), pair[5]),\
                      ((float(pair[i]) - mini[i-1]) / (maxi[i-1] - mini[i-1]) - query_96_normalized[i])**2

    # the combiner produces a pair for each sample as ((ID, label), so-far sum of squared differences)
    def combiner_get_EuDis(self, id_label, squared_diff):
            yield id_label, sum(squared_diff)

    # this reducer produces a pair for each sample as (Euclidean distance from query, (ID, label))
    def reducer_get_EuDis(self, id_label, sumSoFar_squared_diff):
            yield None, (sqrt(sum(sumSoFar_squared_diff)), id_label)

    # this reducer finds the 15 nearest pairs to the query as (distance, (ID, label)) and
    # creates a list of labels of the 15 nearest samples to the query
    # yields the most frequent label of the 15 nearest samples, (query ID, predicted label)
    def reducer_find_nearestSamples(self, _, dis_pair):
        id_label_pair = list()
        for idLabel_distance in heapq.nsmallest(15, dis_pair):
            id_label_pair.append(idLabel_distance[1])
        labels_nearest = [id_label_pair[i][1] for i in range(len(id_label_pair))]
        yield query_96[0], max(set(labels_nearest), key=labels_nearest.count)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_pairs,
                   combiner=self.combiner_get_EuDis,
                   reducer=self.reducer_get_EuDis),
            MRStep(reducer=self.reducer_find_nearestSamples)
        ]


if __name__ == "__main__":
    KNN.run()



