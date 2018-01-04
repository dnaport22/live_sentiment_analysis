from google.cloud import vision
from google.cloud.vision import types

class SentimentAnalysis():

    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    def __init__(self, image):
        self.image = image

    def get_sentiments(self):
        client = vision.ImageAnnotatorClient()
        image = types.Image(content=self.image)
        response = client.face_detection(image=image)
        labels = response.face_annotations
        sentiments = {}
        for label in labels:
            sentiments = {
                'JOY': SentimentAnalysis.likelihood_name[label.joy_likelihood],
                'SORROW': SentimentAnalysis.likelihood_name[label.sorrow_likelihood],
                'SURPRISE': SentimentAnalysis.likelihood_name[label.surprise_likelihood],
                'ANGER': SentimentAnalysis.likelihood_name[label.surprise_likelihood]
            }

        return sentiments