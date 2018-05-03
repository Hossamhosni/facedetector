from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect
from django import forms
from . import recognizer
import cv2
import glob
# Create your views here.

def index(request):
    return render(request, 'emotiondetector/home.html')

def webcam(request):
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("testingImage.jpg", gray)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    request.url = '/'
    uploaded_file_url = "testingImage.jpg"
    prediction = prediction = recognizer.predict("testingImage.jpg")
    return render(request, 'emotiondetector/detector.html', {
        'uploaded_file_url': uploaded_file_url,
        'prediction': prediction
    })

def detect(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        prediction = recognizer.predict(filename)
        prediction = prediction.capitalize()
        #fs.delete(filename)
        redirect('/webcam')
        return render(request, 'emotiondetector/detector.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': prediction,
            'img': myfile.name
        })
    return render(request, 'emotiondetector/detector.html')
