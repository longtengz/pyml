def hamming_distance(sA, sB):
    """Hamming distance between two strings
    of equal length is the number of positions
    at which the corresponding symbols are different.
    """
    if len(sA) != len(sB):
        raise ValueError('Sequences must be of equal length to compute hamming distance')
    return sum(ele1 != ele2 for ele1, ele2 in zip(sA, sB))



def euclidean_distance(sA, sB):
    if len(sA) != len(sB):
        raise ValueError('Sequences must be of equal length to compute euclidean distance')

    sum = 0

    for index, element in enumerate(zip(sA, sB)):
        sum += (element[0] - element[1]) ** 2

    return sum ** 0.5
