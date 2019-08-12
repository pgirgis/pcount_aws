# pcount_aws
A People Counter Application 

Developed by Peter Girgis, pgirgis@bigmate.com.au for use by all.

Modified from good work done by Adrian Rosebrock https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

Designed for python3

1. Follow the link above to install the dependencies such as imutils.
2. Add a new AWS IoT Device 
3. Add the certificates to the certs folder (the ones there are empty placeholders)
4. Configure the config.json to point ot the certificates and AWS IoT Endpoint
5. Configure any other config.json settings
6. Run python3 people_counter.py

Some notes:
- The video save is currently not working.  It also slows the capture process down significantly so I don't recommed using it
- Images are transferred to AWS IoT as base64 encoded images.  This means all images are available in AWS IoT.  You can increase the quality simply by increasing the value in line 426 to make the width bigger pushFrame = imutils.resize(frame,width=40). 200 is good and should stay within the 128kb IoT limits but give a really clear picture
- This version only supports Caffe and MobileNet SSD.  Some code is already in there to handle others. YOLO is really good and fast by comparison
- Uses the horizontal line in the middle of the frame as the counter. 
  
