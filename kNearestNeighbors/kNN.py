import operator

import distance

# distance_func = distance.hamming_distance

distance_func = distance.euclidean_distance

# all arguments must be int-elemented

def kNN(inputData, dataPool, labelPool, k):
    distance = list()

    for data in dataPool:
        distance.append(distance_func(inputData, data))

    sorted_index = sorted(range(len(distance)), key = lambda k: distance[k])

    k_index = sorted_index[:k]

    neighbor_labels = [labelPool[index] for index in k_index]

    # print(neighbor_labels)

    neighbor_labels_count_dict = dict()

    for label in neighbor_labels:
        if label not in neighbor_labels_count_dict:
            neighbor_labels_count_dict[label] = 1
        else:
            neighbor_labels_count_dict[label] += 1

    # print(neighbor_labels_count_dict)

    sorted_dict = sorted(neighbor_labels_count_dict.items(), key = operator.itemgetter(1))

    # print(sorted_dict)

    return sorted_dict[-1][0]
