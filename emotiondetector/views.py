from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from . import recognizer
import cv2
import glob
# Create your views here.

def index(request):
    return render(request, 'webapp/home.html')

def detect(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        prediction = recognizer.predict(filename)
        #fs.delete(filename)
        return render(request, 'webapp/detector.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': prediction,
            'img': filename
        })
    return render(request, 'webapp/detector.html')
