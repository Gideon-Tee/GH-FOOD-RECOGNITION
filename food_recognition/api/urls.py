from django.urls import path
from .views import FoodRecognitionView
from . import views

urlpatterns = [
    path('recognize', FoodRecognitionView.as_view(), name='food-recognition')
]