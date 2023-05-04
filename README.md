
## The Problem

We would like you to write an img2img algorithm that creates a neural network that can automatically derive a **Sobel Filter Kernel** based on a set of input and output images. The output images are simply the input images with a sobel filter applied.

You should make the decision of what kind of network to use, how to structure the layers, etc. Most architectures will work as the network is likely to very quickly converge, but keep in mind what goal you are optimizing for.

**NOTE:** if your model architecture is very deep or you are borrowing an off-the-shelf model, you are likely not thinking about this problem in the right way. 

## Some questions

1) What if the image is really large or not of a standard size?

Answer: There are several ways to handle this. We can either crop some of our images to have all our inputs be the same size, yet, this would be far from ideal as we would lose significant information. A better way to handle this would be to perform tiling on our images and train the tiles parts of one input in series. This would be similar to truncation in text-based models.

2) What should occur at the edges of the image?

Answer: We would have to apply some sort of padding at the edges to our images in order to make sure all images inputted and outputed are of the same size. We can use zero padding, which would basically put bunch of zeros on the edges so that each image has the same size. We can also handle this with replication or reflection padding strategies.

3) Are you using a fully convolutional architecture?

Answer: Yes, we are using a fully convolutional architecture which, in our case, consists of a single convolutional layer with 3x3 filter.

4) Are there optimizations built into your framework of choice (e.g. Pytorch) that can make this fast?

Answer: Yes, PyTorch overall has pretty efficient memory management. Given the use of Nvidia's CUDA library in PyTorch (or at least the ability to use CUDA features by choice), our framework does pretty well in terms of GPU acceleration. PyTorch also has optimized implementations in regards to performing convolutional operations. Although not specific to PyTorch, playing around with hyperparameters like learning rate, batch size, and steps for gradient accumulation would also help increase our traning speed. Again, the last part is not really used

5) What if you wanted to optimize specifically for model size?

Answer: Given that our model is basically a single convolutional layer, there is not much we can do in terms of reducing the number of parameters. Yet, post-training quantization like switching to 8-bit integers or 16-bit floating points from the default 32-bit floating point representations of model weights can certainly speed up our process. Again, since we only have one convolutional layer, our model is very small, thus, we have not a lot of optimization we can perform. If the model was larger, we could have experimented with adapter layers as well as some regularization techniques, yet, such implementations would not be necessarily applicable in our situation.

6) How do you know when training is complete?

Answer: We have to look into our training loss and look for a case of convergence. We can also evaluate our validation loss the same way or perform early stopping depending on how our model improves over certain number of epochs.

7) What is the benefit of a deeper model in this case? When might there be a benefit for a deeper model (not with a sobel kernel but generally when thinking about image to image transformations)?

Answer: In the case of implementing a Sobel Filter, a deeper model is certainly not necessary as our single-layered CNN can perform the desired operations. Yet, a deeper model in general would help significantly in terms of feature extracting, context learning, and generalization over large amounts of data. A larger/deeper model would introduce more parameters, which would most likely help with the robustness of our model. If we are looking to perform more complex Computer Vision downstream tasks, a deeper model would certainly allow us to gain more information on our inputs and desired outputs.

## What is a Sobel Filter?

[Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator) has everything you need to know to answer this question. 

## What libraries should I use?

Pytorch is our favorite, but it's up to you! For a Sobel filter that you can use to construct the training set, there are many options (including just doing the convolution in torch), but there's the `cv2.Sobel` filter pre-made. 

## Where can I get a training dataset?

This is up to you! There are lots of easy dataset libraries out there. Often these libraries allow you to write in transforms or other easy operators. You can always just download COCO or ImageNet. There's of course [img2dataset](https://github.com/rom1504/img2dataset)

## Extra credit question

Now generalize your algorithm so that it can learn any arbitrary image kernel-based filter. You could test this by randomizing the kernel. What are the limitations of this?
