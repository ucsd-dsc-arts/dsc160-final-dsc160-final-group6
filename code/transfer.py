import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np
import time
import imageio
from IPython.display import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

#get base pics in base folder
basepics = []
for i in os.listdir('./images/bases'):
    if (i != '.ipynb_checkpoints'):
        basepics.append(i)

#get style pics in style folder
stylepics = []
for i in os.listdir('./images/styles'):
    if (i != '.ipynb_checkpoints'):
        stylepics.append(i)

#go through every base picture
for base in basepics:
    #apply every style to it
    for style in stylepics:
        
        print('Starting -- base pic: '+base+' style pic: '+style)
        
        content_file_name = base #base pic
        style_file_name = style #style pic

        #name of pics
        base_name = base[0:base.index('.')]
        style_name = style[0:style.index('.')]

        # number of iterations for optimization. 
        iterations = 20 

        #weights of the different loss components
        total_variation_weight = 1.0 
        style_weight = 1.0 
        content_weight = 0.025

        #get image paths
        base_image_path = './images/bases/'+content_file_name
        style_reference_image_path = './images/styles/'+style_file_name


        # get dimensions (width, height) of the generated picture
        width, height = load_img(base_image_path).size

        #rescale images
        img_nrows = 400 
        img_ncols = int(width * img_nrows / height)

        # pre-process the image: rescaling, running it through VGG19

        def preprocess_image(image_path):

            img = load_img(image_path, target_size=(img_nrows, img_ncols))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = vgg19.preprocess_input(img)
            return img


        # uconvert a tensor into a valid image

        def deprocess_image(x):

            if K.image_data_format() == 'channels_first':
                x = x.reshape((3, img_nrows, img_ncols))
                x = x.transpose((1, 2, 0))

            else:

                x = x.reshape((img_nrows, img_ncols, 3))

            # Remove zero-center by mean pixel

            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68

            # 'BGR'->'RGB'

            x = x[:, :, ::-1]
            x = np.clip(x, 0, 255).astype('uint8')
            return x
        

        # get tensor representations of images
        base_image = K.variable(preprocess_image(base_image_path))
        style_reference_image = K.variable(preprocess_image(style_reference_image_path))

        #generated image
        if K.image_data_format() == 'channels_first':
            combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
        else:
            combination_image = K.placeholder((1, img_nrows, img_ncols, 3))


        # combine the style, base, and result images into a single Keras tensor

        input_tensor = K.concatenate([base_image,
                                    style_reference_image,
                                    combination_image], axis=0)

        # build the VGG19 network with our 3 images as input
        model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
        print('Model loaded.')

        # get the symbolic outputs of each "key" layer.
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # compute the neural style loss
      
        # the gram matrix of an image tensor (feature-wise outer product)

        def gram_matrix(x):

            assert K.ndim(x) == 3
            if K.image_data_format() == 'channels_first':
                features = K.batch_flatten(x)
            else:
                features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

            gram = K.dot(features, K.transpose(features))
            return gram

        # the style loss to maintain the style of the reference image in the generated image.

        def style_loss(style, combination):

            assert K.ndim(style) == 3
            assert K.ndim(combination) == 3

            S = gram_matrix(style)
            C = gram_matrix(combination)
            channels = 3
            size = img_nrows * img_ncols
            return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


        # an auxiliary loss function to maintain the "content" of the base image in the generated image

        def content_loss(base, combination):
            return K.sum(K.square(combination - base))


        # total variation loss, keep the generated image locally coherent

        def total_variation_loss(x):
            assert K.ndim(x) == 4
            if K.image_data_format() == 'channels_first':

                a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
                b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
            else:
                a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
                b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
            return K.sum(K.pow(a + b, 1.25))

        # combine these loss functions into a single scalar

        loss = K.variable(0.)
        layer_features = outputs_dict['block5_conv2'] #content features
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(base_image_features,
                                            combination_features)

        # style_features
        feature_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1',
                        'block5_conv1']

        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(feature_layers)) * sl

        loss += total_variation_weight * total_variation_loss(combination_image)

        # get the gradients of the generated image wrt the loss

        grads = K.gradients(loss, combination_image)
        outputs = [loss]

        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([combination_image], outputs)

        def eval_loss_and_grads(x):

            if K.image_data_format() == 'channels_first':
                x = x.reshape((1, 3, img_nrows, img_ncols))
            else:
                x = x.reshape((1, img_nrows, img_ncols, 3))

            outs = f_outputs([x])
            loss_value = outs[0]
            if len(outs[1:]) == 1:
                grad_values = outs[1].flatten().astype('float64')
            else:
                grad_values = np.array(outs[1:]).flatten().astype('float64')
            return loss_value, grad_values



        # Evaluator class to compute loss and gradients in one pass

        class Evaluator(object):

            def __init__(self):
                self.loss_value = None
                self.grads_values = None

            def loss(self, x):
                assert self.loss_value is None
                loss_value, grad_values = eval_loss_and_grads(x)
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        evaluator = Evaluator()

        # run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss
        x = preprocess_image(base_image_path)

        result_prefix ='./images/results_'

        # minimise the loss function

        for i in range(iterations):

            print('Start of iteration', i)
            start_time = time.time()

            #minmimmize loss
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                            fprime=evaluator.grads, maxfun=20)

            print('Current loss value:', min_val)
            
            # save current generated image
            img = deprocess_image(x.copy())
                
            #time to process
            end_time = time.time()

            print('Iteration %d completed in %ds' % (i, end_time - start_time))

        #save final result image in result folder
        imageio.imwrite('./images/results/result_'+base_name+'_'+style_name+'.png', img)
        print(base_name+'_'+style_name+ ' done!')
        print('---------------------------------------')
