class Classification:
    def __init__(self, features, labels, model):
        self.__features = features
        self.__labels = labels
        self.__model = model
        self.__result_label = self.classification()

    def classification(self):
        # Train a random forest classifier on the features and labels
        if not self.__model:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(self.__features, self.__labels)

        # Use the trained classifier to predict the label of the iris image
        label = model.predict(self.__features)
        return label

    def get_result_label(self):
        return self.__result_label
