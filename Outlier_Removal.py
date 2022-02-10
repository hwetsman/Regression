def Outlier_Removal(predictions, training_features, training_labels, N):
    """
        Clean away the N number of points that have the largest
        residual errors and return a list of tuples named cleaned_data where
        each tuple is of the form (new_features, new_labels, error).
	Requires numpy.
    """

    predictions = [x[0] for x in predictions]
    training_labels = [x[0] for x in training_labels]
    error = np.absolute(np.array(predictions) - np.array(training_labels))
    idx = np.argpartition(error, -N)[-N:]
    new_features = np.delete(training_features, idx)
    new_labels = np.delete(training_labels, idx)
    error = np.delete(error, idx)
    cleaned_data = list(zip(new_features, new_labels, error))
    return cleaned_data
