# dog_breed_identification

A comparative study of different Convolution Neural Networks applied on the Kaggle Dog Breed Classification dataset.
This kind of comparative study is usually performed at the beginning of a project to quickly zero-in on a model that can 
get you started in the right direction. Later, the chosen model can be fine-tuned to perform better.

The code has the follwing general structure:
* Load the images and the labels
* Normalize the images and one-hot-encode the labels
* Create a train/val data split
* Create and Train the model (pre-trained on imagenet)
* Visualize the training

### Experiment 1
[dog_breed_pretrained.ipynb](https://github.com/kapil-varshney/dog_breed_identification/blob/master/dog_breed_pretrained.ipynb)
* Models are pretrained on ImageNet dataset
* Training done only for 4 epochs
* The base layer are frozen and only the classification layer is allowed to train

Results:

Model | Loss | Acc | Val Loss | Val Acc
----- | ---- | --- | -------- | -------
VGG16 | 1.9177 | 0.8607 | 6.0462 | 0.2220
VGG19 | 7.0760 | 0.5426 | 9.6538 | 0.1550
ResNet50 | 0.1764 | 0.9632 | 0.9153 | **0.7413**
InceptionV3 | 13.9648 | 0.1293 | 14.1188 | 0.1193
InceptionResNetV2 | 11.8685 | 0.2546 | 12.6438 | 0.2054
Xception | 9.8719 | 0.3764 | 10.7640 | 0.3139

Inference:
ResNet50 seems to perform the best among others. Although, it looks overfitted (Training Acc is high vs Validation Accuracy), 
it still manages to perform decently on the Validation data. VGG16 is highly overfitted while other models just couldn't give 
any good results.

### Experiment 2
[dog_breed_pretrained_with_avgpool_fc.ipynb](https://github.com/kapil-varshney/dog_breed_identification/blob/master/dog_breed_pretrained_with_avgpool_fc.ipynb)
* Models are pretrained on ImageNet dataset
* Added an Average Pooling + FC (Fully Connected) layer over the base models
* For the first 4 epochs the base layers are frozen and only the classification layer is allowed to train
* For the next 25 epochs the complete model is unfrozen and allowed to train

Results:

Model | Loss | Acc | Val Loss | Val Acc
----- | ---- | --- | -------- | -------
VGG16 | 1.2464 | 0.6665 | 2.4761 | 0.3863
VGG19 | 1.7082 | 0.5454 | 2.6956 | 0.3403
ResNet50 | 0.0738 | 0.9754 | 1.9211 | **0.7042**
InceptionV3 | 0.2326 | 0.9258 | 2.2944 | 0.5653
InceptionResNetV2 | 0.2350 | 0.9189 | 1.8065 | 0.6650
Xception | 0.0909 | 0.9719 | 2.0337 | 0.6445

Inference: ResNet50 still seems to be performing better than other models. An interesting observation is that ResNet50 has 
dropped in val acc compared to the previous experiment, while the other models have improved on that metric. 
We'll have to be carefull of overtraining the model which all the models seems to be suffering from.

### Next Steps
ResNet50 seems to be giving a good starting point. We can start to work on improving this model by:
* Data augmentation
* Regularization
* Adaptive Learning
* Better optimizer
