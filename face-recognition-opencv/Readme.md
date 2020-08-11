# This repo allows a user to create a simple face recognition application using a small dataset.


## Required libraries: 
- dlib (installation of dlib requires Cmake to be install into the system)
- face_recognition_python  
- imutils 

First, create your own dataset with faces that needs to be recognized. For that, create a folder named "dataset" in the source directory and 
insert around 10 pictures for every person you need to recognize under seperate folder. |

For example, my dataset folder looks like

    ./dataset/Sakar/1.jpg,2.jpg,3.jpg ...
    ./dataset/Adam/1.jpg,2.jpg,3.jpg ...
   
Then, create a encoding for your current dataset using the provided encode_faces.py script. 
python encode_faces.py --dataset dataset --encodings encodings.pickle 
This will create the encodings.pickle file which will be the saved encodings for every picture in your dataset. 

Then run, recognize_faces_image.py or recognize_faces_video.py file depending upon whether you want to detect faces in a picture or video from your webcam. 
