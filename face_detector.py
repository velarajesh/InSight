from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import os

class Face_Cropper():
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret/creds.json"
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    def _detect_face(self, face_file, max_results=4):
        client = vision.ImageAnnotatorClient()
        content = face_file.read()
        image = types.Image(content=content)

        return client.face_detection(image=image, max_results=max_results).face_annotations


    def process_faces(self, image, faces):
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                           'LIKELY', 'VERY_LIKELY')
        weights = {'UNKNOWN': 0, 'VERY_UNLIKELY': 0, 'UNLIKELY': 0.01, 'POSSIBLE':0.03,
                       'LIKELY': 0.05, 'VERY_LIKELY': 0.06}
        score = dict()
        im = Image.open(image)
        draw = ImageDraw.Draw(im)
        face = faces[0]
        box = [(vertex.x,vertex.y) for vertex in face.bounding_poly.vertices]
        # draw.line(box+[box[0]],width = 6,fill='#00ff00')
        draw.text((face.bounding_poly.vertices[0].x,
                   face.bounding_poly.vertices[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
        #print(box[0],' ',box[2])
        #print((box[0][0],box[0][1],box[2][0],box[2][1]))

        score["Surprise"] = weights[likelihood_name[face.surprise_likelihood]]
        score["Happiness"] = weights[likelihood_name[face.joy_likelihood]]
        score["Anger"] = weights[likelihood_name[face.anger_likelihood]]
        score["Sadness"] =weights[likelihood_name[face.sorrow_likelihood]]

        # print(score)

        im = im.crop((box[0][0], box[0][1], box[2][0], box[2][1]))
        return (im, score)

    def get_cropped_face(self, input_filename, max_results):
        with open(input_filename, 'rb') as image:
            faces = self._detect_face(image, max_results)
            #print('Found {} face{}'.format(
                #len(faces), '' if len(faces) == 1 else 's'))
            #print(faces)
            #print('Writing to file {}'.format(output_filename))
            # Reset the file pointer, so we can read the file again
            image.seek(0)
            return self.process_faces(image, faces)
