"""Basic checks for Flask routes and the two saved models."""
import unittest
from pathlib import Path

from app import app

ROOT = Path(__file__).resolve().parents[1]


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_pages_open(self):
        for url in ["/", "/static/styles.css", "/static/app.js", "/api/report"]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, url)

    def test_models_load(self):
        reply = self.client.get("/api/health")
        self.assertEqual(reply.status_code, 200)
        self.assertTrue(reply.get_json()["models_loaded"])

    def test_saved_results(self):
        result = self.client.get("/api/meta").get_json()
        self.assertEqual(result["total_images"], 1090)
        self.assertEqual(result["selected_model"], "svm")

    def test_example_image(self):
        image = (ROOT / "examples" / "orange_blackspot_example.jpg").read_bytes()
        response = self.client.post("/api/predict", data=image, content_type="image/jpeg")
        self.assertEqual(response.status_code, 200)
        answer = response.get_json()
        self.assertEqual(answer["svm"]["label"], "blackspot")
        self.assertEqual(answer["cnn"]["label"], "blackspot")

    def test_invalid_upload(self):
        response = self.client.post("/api/predict", data=b"not an image", content_type="image/jpeg")
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/predict", data=b"text", content_type="text/plain")
        self.assertEqual(response.status_code, 415)
        response = self.client.post("/api/predict", data=b"x" * (8*1024*1024+1), content_type="image/jpeg")
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.get("/api/plot/unknown").status_code, 404)


if __name__ == "__main__":
    unittest.main()
