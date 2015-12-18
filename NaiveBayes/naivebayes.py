from __future__ import division
import math

# TODO @saved_file saves the conditional probabilities for this bayes net

# @lables is 1*n array

# @features is 'n' * 'number of features', each element is a single 'feature' list that contains arbitrary numbe of values a feature can take on
# so features is a 3-d array

# implement it as features can take on multiple values

class NaiveBayes():

    def __init__(self, smoothing_factor = 0):
        self.smoothing_factor = smoothing_factor

    def train(self, labels, features):
        
        # P( C | F ) = P(C) * P(F|C) / P(F)
        #            = P(C) * P(F1|C) * P(F2|C) * .... * P(Fn|C) / P(F1,F2,,,,Fn)
        # P(F) here is just a constant when given specific features and trying to predict the right label, 
        # because you always go for the one with largest P(C|F), so having P(F) or not doesn't affect the end result



        # P(C=c1) can directly be calculated from the data we have by looping the labels
        #      for example, if we have 8 instances out of 100 instances are labeled as digit '9'
        #      then we say P(C='9') = 8/100

        # compute the self.priors (type: dict)

        totalNumInstances = len(labels)

        self.total_num_trained = totalNumInstances

        self.unique_labels = list(set(labels))

        # {label1: 0, label2: 0, ,,, labeln: 0} though it doesn't necessarily keeps numerical order
        numLabels = dict(zip(self.unique_labels, [0 for i in range(totalNumInstances)])) 

        for label in labels: 
            numLabels[label] = numLabels[label] + 1

        self.priors = numLabels

        for key, value in self.priors.iteritems():
            self.priors[key] = value / totalNumInstances


        # P(Fn=fn1|C=c1) can be calculated by looping through instances by labels,
        #         counting how many times this feature 'Fn' has taken the value 'fn1's in label 'c1'     
        #         and dividing that count by the number of times 'c1' has shown up in the data set


        # compute self.posteriors (type: dict)
        # posterior = {label1: [{feature_1_value_1:xxx, feature_1_value_2:xxx}, {feature_2_value_1:xxx, feature_2_value_2:xxx} ,,,,,,]}

        # {label1: [{}, {}, ,,,{}], label2: [{}, {}, ,,,{}], ,,, labeln: [{}, {}, ,,,{}]}
        self.posteriors = dict(zip(self.unique_labels, [[{} for z in range(len(features[0]))] for x in range(totalNumInstances)]))

        # 'n' * 'number of features'
        # @feature_vec is [feature_1_value, feature_2_value,,,,,feature_n_value]
        for index, feature_vec in enumerate(features):
            # first round of looping, just do the counting
            label = labels[index]
            for feature_idx, feature_val in enumerate(feature_vec):
                if feature_val in self.posteriors[label][feature_idx]:
                    # increase the count
                    self.posteriors[label][feature_idx][feature_val] = self.posteriors[label][feature_idx][feature_val] + 1
                else:
                    # if it's a new value never seen before for this feature
                    self.posteriors[label][feature_idx].update({feature_val: 1})

        # compute posteriors using the count we compute in the previous for loop
        for label, value in self.posteriors.iteritems():
            for feature_dict in value:
                feature_value_total_count = sum(feature_dict.values())
                # feature_value_total_count = len(labels)
                for feature_value, feature_value_count in feature_dict.iteritems():
                    # using laplace smoothing
                    posterior = (feature_value_count + self.smoothing_factor) / (feature_value_total_count + self.smoothing_factor * len(features[0]))
                    feature_dict[feature_value] = posterior




    def classify(self, features):
    
        # @features is 'n' * 'num of features' list

        labels = []
        probability = 0
        max_probability = 0
        predicted_label = None

        # for every instance needs predicting
        for index, feature_vec in enumerate(features):

            # compute the probabilty and argmax to get the predicted label
            for idx, label in enumerate(self.unique_labels):

                # using log to prevent underflow since so many decimals are multilying 
                probability = math.log(self.priors[label])

                for feature_idx, feature_val in enumerate(feature_vec):
                    # TODO is this correct?
                    if feature_val in self.posteriors[label][feature_idx]:
                        posterior = math.log(self.posteriors[label][feature_idx][feature_val])
                    else:
                        if self.smoothing_factor != 0:
                            # this is just a very coarse estimation of the probability of this unseen feature given the label
                            # TODO note the "plus 1" part, it needs justification 
                            posterior = math.log(self.smoothing_factor / (self.total_num_trained + self.smoothing_factor * (1 + len(feature_vec))))
                        else:
                            # this means even if one single difference in the features between two extremely similar instances will fail classify them as the same
                            posterior = 0

                    probability = probability + posterior

                #print probability

                if idx == 0:
                    max_probability = probability
                    predicted_label = label

                if probability > max_probability:
                    max_probability = probability
                    predicted_label = label

            # store the predicted label
            labels.append(predicted_label)

        
        return labels

















