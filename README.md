# Object-Detection

Car object detection from a video has been implemented and then the classification whether the car is an SUV and sedan has been implemented. The project haas been implemented in a pipeline where Q1 is creating frame and is called the producer and Q2 is the consumer which is detecting the object in the frame and classifying whetehr it is a SUV or sedan. 

The producer consumer pipeline has resulted in icreasing the throughput from 349 secs to 270 secs![image](https://user-images.githubusercontent.com/63654651/117534616-243b3480-afea-11eb-94bc-ea2baf4b70f2.png)
Tiny-yoloV3 model has been used for object detection and mobilenet classifier has been user for the classification.

**Task1 - To count number of car in a frame :**<br>
F1 score of 88 % was achieved

**Task2 - To count SUV and SEDAN car in a frame** :<br>
F1 score of 87 % was achieved

## Classifier Model 
For the classification of the car type for the detected car, MobileNet architecture has been used as our base model. These are built by depthwise separable convolutions. MobileNet model has been trained on the coco dataset and we are using transfer learning to train our deep learning models.
Transfer learning has been utilized in the classification of the car labels. We leverage the features that were learned at the training of the base model. We simply use the pre-trained model because of its faster speed. We are tuning the model by adapting the model to the new input – output pair of our interest which is car type. Various steps involved in building the classification model:
1) Preparation and Pre-processing of Dataset for training
The Car dataset from Stanford has been used for the training of the classifier model. Link has been provided in the references. <br>
● The images for the sedan and SUV types are extracted. 500 images of each SUV and sedan class are used for creation of the dataset and labels of each class are provided for the supervised learning. <br>
● Each image in the dataset is resized to (224,224) as the input shape required for the MobileNet model is (224,224). <br>
● The images are normalized.<br>
2) Creation of train and test sets
The normalized images are then split into train and test datasets in the ratio of 70:30.
3) Initialization of the base model – A base model using MobileNet V2 is created which is a pretrained model.
4) Classification model – Since the pre-trained model is trained on coco dataset which can classify 1000 classes so we will manually set the out layers to classify 2 classes. For the purpose, a single node in the output layer is considered as we are dealing with a binary classification. While using pre-trained model for classification, the initial layers of the model architecture has been frozen and we are only changing the last layer which has to detect the two classes i.e., “sedan” and “SUV”.
