# SDCND_Traffic_sign_classifier

Rubric points:

#### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of each traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset
 I loaded the "signnames.csv" using csv library and print a random image with it´s label

### Design and Test a Model Architecture

1. The data process follows below structure:
* Convert the image into greyscale
* Normalize the image to have zero mean values and equal variance. First I used below code:
  
def normalize(imagen):
greyscale=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  
normalized=np.empty([32, 32, 1], dtype=float)  
for i in range(32):
for j in range(32):
normalized[i][j]=float((greyscale[i][j]-128)/128) 
return normalized
  
  later I realized there is a function that already performs this operation (cv2.normalize)
*  Equalize image  
2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU	activ. fucntion				|												|
| Max pooling	      	| 2x2 stride,  valid padding,  outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6 	|
| RELU	activ. fucntion				|			
| Max pooling	      	| 2x2 stride,  valid padding,  outputs 5x5x16 	|
| Flatten				|
| Fully connected		|         							outputs 120		|
| RELU	activ. fucntion				|		
| Dropout				|	0.5 |	
| Fully connected		|         							outputs 84		|
| RELU	activ. fucntion				|		
| Fully connected		|         							outputs 43		|	

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The hyperparameters used to train my model are:
* EPOCHS = 40
* BATCH_SIZE = 128
* mu = 0
* sigma = 0.01
* dropout=0.5
* rate=0.001

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Training set accuracy of 97.3%
* Validation set accuracy of 93%
* Test set accuracy of 88.8%

The proccess I followed was really trial and error 
First architecture choosen for the traffic sign classifier was the one from letnet, just modified to have (32,32,3) inputs images. No preprocces was applied to the images. learning rate was set to 0.001. The validation accuracy obtained with that model using 150 EPOCHS was lower than the needed 93%, so I started processing images before modifiying the model structure. 
* 1st) Convert images into grey scale and normalizing to have zero mean and equal standard deviation. Using 150 EPOCHS and 128 batch size got an accuracy of 90%
* 2nd) Use Histogram equalization tool in order to enhance low contrast images. Reducing the number of EPOCHS I was able to reach 91% accuracy
* 3rd) Add 0.5 droput after the 3rd layer. This change led me to get the needed 93% accuracy


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

![Alt text](C:/Users/german/Desktop/1.jpg?raw=true "Title")

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|