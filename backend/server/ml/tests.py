from django.test import TestCase
from ml.classifier.random_forest import RandomForestClassifier
from ml.registry import MLRegistry

class MLRegistryTests(TestCase):
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Kami"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                            algorithm_status, algorithm_version, algorithm_owner,
                            algorithm_description)
        print(registry)
        self.assertEqual(len(registry.endpoints), 1)
