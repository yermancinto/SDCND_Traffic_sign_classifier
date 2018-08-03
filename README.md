# SDCND_Traffic_sign_classifier

Rubric points:

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of each traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

See below the histograms for the trainning, validation and test datasets respectively:
![imagen](https://user-images.githubusercontent.com/41348711/43163666-197b21d2-8f8f-11e8-87a3-e42136425f70.png)
There are hudge differences in the naumber of samples of each traffic sign. Logic tells me that the more numerous, better trained the model for that specific traffic sign, so in the next step. Less numerous classes will (...) 

#### 2. Include an exploratory visualization of the dataset
I added some lines of code to plot 5 signals of each type so I can understand better the handicap my NN has to deal with. It is a slow process so I just showed below 5 traffic sign samples that summarize the whole dataset. The complete image matrix can be found under the *.html file.   

50kph speed limit:
![imagen](https://user-images.githubusercontent.com/41348711/43152871-038a6410-8f70-11e8-93b9-66204018ef3e.png)
60kph speed limit: (even if you can not see the last two images are there...)
![imagen](https://user-images.githubusercontent.com/41348711/43153037-759d6b42-8f70-11e8-83ee-b973d7e882fc.png)
Bicycles crossing:
![imagen](https://user-images.githubusercontent.com/41348711/43153653-2734be36-8f72-11e8-9cbc-8053fa48598b.png)
Be aware (ice/snow):
![imagen](https://user-images.githubusercontent.com/41348711/43153788-8b521134-8f72-11e8-9e2d-5bfec22c9f52.png)
Turn left ahead:
![imagen](https://user-images.githubusercontent.com/41348711/43154025-39a854d2-8f73-11e8-8f03-d36b40882c6a.png)

From this first exploration we can extract the main problems our model will face:
* High brightness images
* Low contrast images
* Blurred images
* Signs covered by trees or other objects
* Different sizing
* Some of them are in perspective view or slightly rotated
* Pieces of other traffic signs in the same picture

### Design and Test a Model Architecture

#### 1. Augment the existing data Set:

As the Trainning Data Set Histogram shows high differences in the number of samples, modified images are generated using the existing ones as a base. To do so , my code stores the images for every class in a temporal array and then runs a function to randomly generate modified images from the existing ones. The code just generates the images needed to match the larger class (in this case the class no.2 -> Speed limit 50kph)

The function follow the different steps: 

1. Get a random image from previously mentioned array.
2. Apply a 3x3 Kernel 0-Standard variation Gaussian Blurring. Based on some trials and errors.
2. Apply a random translation. Just up to +-2 pixels on X and Y axis. Those 2 pixels came as a result of some images results.
3. Apply a random image rotation (uo to +-6 degrees) using cv2.getRotationMatrix2D tool. A higher angles results on too artificial rotations
4. Apply a perspective view transformation using cv2.getAffineTransform. Parameters were tunned based on experience.

At the end of this process, 51631 images are generated and stored as X_Train Data Set

Extra training data set is concatenated with the existing one and whole datset is pickled. I did not how to pickle data so for this part I copied the code from internet


#### 2. Process the images:

1. The data process follows below structure:
* Convert the image into greyscale (cv2.cvtColor)
* Normalize the image to have zero mean values and equal variance (cv2.normalize)
* Apply CLAHE histogram Equalization (cv2.createCLAHE)

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
* EPOCHS = 120
* BATCH_SIZE = 128
* mu = 0
* sigma = 0.01
* dropout=0.7
* rate=0.0001  (low learning rate but better convergence)

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Training set accuracy of 99.6%
* Validation set accuracy of 95.0%
* Test set accuracy of 92.0%

The proccess I followed was really trial and error 
First architecture choosen for the traffic sign classifier was the one from letnet, just modified to have (32,32,3) inputs images. No preprocces was applied to the images. learning rate was set to 0.001. The validation accuracy obtained with that model using 150 EPOCHS was lower than the needed 93%, so I started processing images before modifiying the model structure. 
* 1st) Convert images into grey scale and normalizing to have zero mean and equal standard deviation. Using 150 EPOCHS and 128 batch size got an accuracy of 90%
* 2nd) Use Histogram equalization tool in order to enhance low contrast images. Reducing the number of EPOCHS I was able to reach 91% accuracy
* 3rd) Add 0.5 droput after the 3rd layer. This change led me to get the needed 93% accuracy
* 4th) Using the recommendations from reviser I augmented the data set and change the equalization to a CLAHE histogram equalization. This way I got 95% accuracy for the validation data set

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Picture below shows six images randomly selected from the german traffic sign dataset and the prediction results using my final model: (http://benchmark.ini.rub.de/)

![imagen](https://user-images.githubusercontent.com/41348711/43659893-a69a4c2a-975d-11e8-8ea2-aee86bc0764b.png)

For this particular case I aws lucky and the result is 100% accuracy

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

Except for the second image, the NN identifies easily the traffic signs. It is possible that for the second image, the white spot beside the arrow confuses the NN giving a 28% probability to "No passing" traffic sign 

See below detailed predictions: 

1st image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 1-Speed limit 30kph 									| 
| .00     				| 2-Speed limit 50kph									|
| .00					| 7-Spped limit 100kph											|
| .00	      			| 5-Speed limit 80kph			 				|
| .00				    | 0-Speed limit 20kph      							|

2nd image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| 35-Ahead only							| 
| .28     				| 9-No passing	 										|
| .01					| 3-Spped limit 60kph											|
| .00	      			| 10-No passing for vehicles over 3.5 metric tons 				|
| .00				    | 15-No vehicles	     							|


3rd image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| 11-Right-of-way at the next intersection							| 
| .00     				| 36-Go straight or right 										|
| .00					| 12-Priority road											|
| .00	      			| 40-Roundabout mandatory			 				|
| .00				    | 30-Beware of ice/snow	     							|

4th image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 10-No passing for vehicles over 3.5 metric tons							| 
| .00     				| 9-No passing 										|
| .00					| 15-No vehicles											|
| .00	      			| 3-Speed limit 60kph			 				|
| .00				    | 5-Speed limit 80kph	     							|

5th image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| 13-Yield						| 
| .00     				| 35-Ahead only 										|
| .00					| 15-No vehicles											|
| .00	      			| 3-Speed limit 60kph			 				|
| .00				    | 9-No passing	     							|

6th image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| 15-No vehicles					| 
| .00     				| 8-Speed limit 120kph										|
| .00					| 12-Priority road										|
| .00	      			| 4-Speed limit 70kph			 				|
| .00				    | 32-End of all speed and passing limits	     							|
