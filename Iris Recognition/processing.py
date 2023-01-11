class Processing:
    def __init__(self, img):
        self.__img = img
        self.__iris = self.preprocessing()

    def preprocessing(self):
        # Convert image to grayscale
        gray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)

        # Detect the iris region
        iris_cascade = cv2.CascadeClassifier('path_to_iris_cascade.xml')
        iris_rect = iris_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Crop the iris region and normalize the size
        x, y, w, h = iris_rect[0]
        iris = gray[y:y + h, x:x + w]
        iris = cv2.resize(iris, (256, 256))

        return iris

    def get_iris(self):
        return self.__iris

