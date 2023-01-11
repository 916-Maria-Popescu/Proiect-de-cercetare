class Extraction:
    def __init__(self, img):
        self.__img = img
        self.__features = self.feature_extraction()

    def feature_extraction(self):
        # Load a pre-trained CNN model
        model = keras.models.load_model('path_to_model.h5')

        # Reshape the image to match the input shape of the CNN
        img = self.__img.reshape(1, 256, 256, 1)

        # Extract features from the iris image using the CNN
        features = model.predict(img)

        return features

    def get_features(self):
        return self.__features
