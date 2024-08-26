# api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
import numpy as np
from PIL import Image
import tensorflow as tf

class FoodRecognitionView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = tf.lite.Interpreter(model_path="/assets/final_best_float32.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image):
        image = image.resize((224, 224)) 
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            image = Image.open(image)
            input_data = self.preprocess_image(image)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(output_data, axis=1)[0]

            # Map the prediction to the class label
            class_labels = [
                'banku',
                'beans',
                'bread',
                'egg-and-pepper',
                'fufu',
                'hausa-koko',
                'jollof',
                'kelewele',
                'kenkey',
                'kokonte',
                'koose',
                'plain-rice',
                'plantain',
                'waakye',
                'yam'
            ]  
            predicted_label = class_labels[predicted_class]

            return Response({'predicted_label': predicted_label}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
