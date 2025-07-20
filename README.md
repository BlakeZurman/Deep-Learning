Ship Image Classification - Deep Learning Project Report
Blake Zurman
IST 691 - Deep Learning
Professor: Mohammed A. Syed
Project Overview
For this project, I built a deep learning model that can look at a photo of a ship and tell whether it’s a Cargo ship, Military ship, Carrier, Cruise ship, or Tanker. This type of image classification is a good example of how deep learning can solve real-world analytics problems. I used this opportunity to apply a bunch of concepts I learned throughout the course.
The Dataset
The data came from a hackathon hosted by Analytics Vidhya and was available on Kaggle. It included training images, a test set, and CSV files with labels. I loaded everything into pandas, created full image paths, and mapped the numeric class labels to their actual ship type names.
Project Steps
1. Data Collection and Setup
I started by organizing the image paths and labels, making sure everything was ready to feed into the model.
2. Exploratory Data Analysis
I plotted the class distribution to see if the dataset was balanced. As you can see, there were many more cargo ship images than the others. I decided to try and train the model without addressing this because for one, I knew I would be using ‘adam’ which can sometimes handle class imbalances, and two, I would be using VGG16. I also looked at some sample images to get an idea of the data quality and variety.
  


3. Preprocessing and Augmentation
To prepare the data for training, I normalized pixel values and used image augmentation like shearing, zooming, and flipping to create more variety. The point of this is to prevent over fitting. I split the data into training and validation sets and resized all the images to the same size.
Train Data 
Test Data 





4. Building a Simple CNN
I built a basic convolutional neural network using a few Conv2D (using relu) and MaxPooling layers, followed by dense layers to do the classification. The final layer used softmax to pick the most likely ship type. I trained it for 10 epochs using the Adam optimizer.
 
5. Using Transfer Learning with VGG16
To improve performance, I tried transfer learning. I used VGG16, which is already trained on a large image dataset. I kept its original layers frozen and added new layers for our ship categories. I trained this model the same way as the first one.
 


6. Evaluation
After training both models, I tested them using the validation set. I recorded the accuracy and loss so I could compare their performance.
How was Overfitting Avoided?
The code uses a separate validation set to check how well the model is doing on data it hasn’t seen during training. After each training round, it tracks both loss and accuracy on this validation set. This makes it easy to spot overfitting, for example:
If the model is doing great on the training data but starts to do worse on the validation set, it’s likely just memorizing rather than learning. Since these validation metrics are saved in the history object, you can review them later to see how well the model is generalizing. This step is important because it helps confirm whether the model will perform well on new data, not just the training set.
CNN Model Interpretation
The CNN model went through 10 epochs of training, and the accuracy and loss improved over time. Here’s a breakdown of what happened:
•	Epoch 1:
The model began with a low training accuracy of around 32% and a high loss. Validation accuracy was around 55%, which already showed some potential.
•	Training Stability:
Early on, there was a warning about running out of data, likely due to not setting steps_per_epoch correctly or not repeating the dataset. That may have caused Epoch 2 to be cut short, so its result (59% accuracy) might not be reliable.
•	Progress Over Epochs:
Over the next few epochs, both training and validation accuracy steadily improved.
o	By Epoch 5, the model reached 66% validation accuracy with reduced loss.
o	By Epoch 9, the model hit a peak validation accuracy of 71.2%, showing clear learning progress.
o	At Epoch 10, validation accuracy slightly increased again to ~72%, with a validation loss of 0.77, which is a strong improvement from where it started.
•	Final Evaluation:
After training, the model’s final accuracy on the validation set was about 71%, with a loss under 0.8, which is a solid result for a basic CNN trained from scratch.
What This Means
•	The model clearly learned how to distinguish between ship types, and the upward trend in accuracy shows it was generalizing better with each epoch.
•	Validation accuracy being close to training accuracy (both in the low 70s) suggests the model isn’t overfitting too much, which is a good sign.
•	The use of data augmentation likely helped the model generalize better and avoid getting stuck memorizing the training set.
Possible Improvements
•	Fixing the dataset pipeline to avoid the early data exhaustion warning would give more reliable training.
•	Adding regularization like dropout, using more data, or training for a few more epochs could push accuracy higher.
VGG16 Model Interpretation
This version of the model used transfer learning by building on top of VGG16, which was pre-trained on a large image dataset. The idea was to use its learned features as a starting point and just train a few custom layers for this ship classification task. The results were strong right from the start and got better across the 10 training epochs.
How It Performed
•	Epoch 1:
Even in the first epoch, the model started with a training accuracy of 55% and a much higher validation accuracy of 77%, which shows the power of using pre-trained features. Validation loss was already low at 0.56.
•	Rapid Improvement (Epochs 2 to 5):
By just the second epoch, training accuracy jumped to 81% and continued rising. Validation accuracy reached 85% by epoch 5, and the validation loss dropped to around 0.39, showing steady improvement.
•	Consistent Results (Epochs 6 to 10):
From epoch 6 onward, both training and validation accuracy stayed strong. Training accuracy reached over 90%, and validation accuracy peaked at about 86.5%. The validation loss stayed around 0.36 to 0.40, which is solid and indicates stability.
•	Results:
After training, the final validation accuracy was about 86%, and the validation loss was 0.40. On the test set, the model achieved 87% accuracy, confirming that it generalized well.
What This Tells Us
•	Transfer learning with VGG16 clearly outperformed the basic CNN model.
•	The model learned quickly and didn't overfit, which shows that the pre-trained features were a great fit for this problem.
•	The steady validation accuracy and loss suggest that the model reached a good balance between learning and generalizing.
Compared to the CNN Model
•	The CNN model topped out at around 72% accuracy, while VGG16 reached 86-87%.
•	Training time was longer … way, way longer… with VGG16, but the results were more reliable and accurate.
•	The CNN model improved slowly over epochs, but VGG16 was strong from the beginning and needed less tuning to get good results.
Using VGG16 gave a major boost in performance and showed how powerful transfer learning can be when you’re working with a small or mid-sized dataset. It saved time and effort while still giving high-quality results. If I continued this project, I’d try ResNet or EfficientNet to see if performance improves even more.

7. How Course Topics Helped Me Build This
This project helped me put a lot of course topics into practice:
•	Deep Learning Basics
I used everything from building a model to choosing activation functions and loss functions.
•	Backpropagation and Optimization
During training, I used backpropagation and optimizers like Adam, just like we learned in class.
•	CNN Fundamentals
The simple CNN model helped reinforce how convolution and pooling layers work to extract features from images.
•	Advanced CNN Topics
I used data augmentation to improve generalization and avoid overfitting, which ties into the regularization methods we talked about.
•	Transfer Learning
The second model used a pre-trained VGG16, showing how transfer learning can improve accuracy with less training time.
8. What I Learned
•	How to handle and prepare image data using TensorFlow and pandas
•	How to design and train CNNs from scratch
•	How transfer learning can help boost performance
•	How choices like batch size, model depth, and augmentation affect results
9. What’s Next
If I had more time, I’d try:
•	Testing other pre-trained models like ResNet or EfficientNet
•	Using more advanced augmentation strategies
•	Fix Class Imbalance
•	Fine-tuning hyperparameters like learning rate and number of layers
•	Looking into model explainability to see what the model focuses on when making predictions
<img width="468" height="449" alt="image" src="https://github.com/user-attachments/assets/adc345d0-8419-4764-9a10-1979b574851d" />
