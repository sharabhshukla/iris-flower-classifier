import unittest
from irisflower.classifier import IrisFlowerClassifier
from sklearn.datasets.base import Bunch
from sklearn.neighbors import KNeighborsClassifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.iris_flower = IrisFlowerClassifier()

    def test_load_dataset(self):
        self.assertIsInstance(self.iris_flower.load_dataset(), Bunch)

    def test_get_trained_data(self):
        dataset = self.iris_flower.load_dataset()
        self.assertIsInstance(self.iris_flower.get_trained_model(dataset.data, dataset.target), KNeighborsClassifier)

    def test_classify(self):
        self.assertIn(self.iris_flower.classify(sepal_length=5,sepal_width=3,petal_length=1,petal_width=0.2), ['setosa', 'versicolor', 'virginica'])

    def test_get_knn_classifier(self):
        self.assertIsInstance(self.iris_flower.get_knn_classifier(), KNeighborsClassifier)

    def test_get_species_by_index(self):
        self.assertEqual(self.iris_flower.get_species_by_index(0), 'setosa')
        self.assertEqual(self.iris_flower.get_species_by_index(1), 'versicolor')
        self.assertEqual(self.iris_flower.get_species_by_index(2), 'virginica')

if __name__ == '__main__':
    unittest.main()
