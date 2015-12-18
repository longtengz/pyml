
# the number of weight vectors equals to the number of distinct labels
class Perceptron():

    # train
    # initialize weight vectors as zeros
    # run the features through each weight vectors, if the weight vector that has the largest activation is not the weight vector corresponding to the correct label, then you subtract that weight vector by the feature and add the correct weight vector by the feature. Otherwise, do nothing
    def train(self, labels, features):

        self.unique_labels = list(set(labels))

        # {label_1:[weight_1_value, weight_2_value,,,weight_n_value], label_2:[],,,,,label_n:[]}
        # the number of weights equals to the number of features in each instance
        # add bias term to be the first element in the weight_vec
        # and initialize all the weights to 0
        self.weights = dict(zip(self.unique_labels, [[0 for z in range(len(features[0]) + 1)] for x in range(len(labels))]))

        # add a feature for the bias term and that featurer has value 1
        for feature_vec in features:
            feature_vec.insert(0, 1)


        for index, label in enumerate(labels):

            idx = 0
            predicted_label = None
            max_activation = 0

            for weight_label, weight_vec in self.weights.iteritems():

                activation = sum([i*j for (i,j) in zip(weight_vec, features[index])])

                if idx == 0:
                    predicted_label = weight_label
                    max_activation = activation

                idx = idx + 1

                if activation > max_activation:
                    predicted_label = weight_label
                    max_activation = activation

            if predicted_label != label:
                # update the weights
                self.weights[predicted_label] = [i-j for i,j in zip(self.weights[predicted_label], features[index])]

                self.weights[label] = [i+j for i,j in zip(self.weights[label], features[index])]
                 

    # classify
    # run the features through each weight vectors, choose the corresbonding label that has the largest activation
    def classify(self, features):

        labels = []

        # add a feature for the bias term and that featurer has value 1
        for feature_vec in features:
            feature_vec.insert(0, 1)

        for feature_vec in features:

            idx = 0
            predicted_label = None
            max_activation = 0

            for weight_label, weight_vec in self.weights.iteritems():

                activation = sum([i*j for (i,j) in zip(weight_vec, feature_vec)])

                if idx == 0:
                    predicted_label = weight_label
                    max_activation = activation

                idx = idx + 1

                if activation > max_activation:
                    predicted_label = weight_label
                    max_activation = activation

            labels.append(predicted_label)

        return labels






