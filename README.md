# Through the Covid-19 Lens

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Iman Nematollahi, imnemato@ucsd.edu
- Soon Gi Shin, sgs008@ucsd.edu
- Justin Lee, jul290@ucsd.edu
- Jaskaranpal Singh, jas137@ucsd.edu
- Dan Ngo, dmngo@ucsd.edu

## Abstract
Since the rise of the COVID-19 pandemic, there has been a significant change in the way that many people view the world. Though there are those who believe that the pandemic is nothing but a hoax, a greater majority has become extremely fearful of the potential threat that the coronavirus poses. Being in quarantine has exacerbated much of these fears as well, as many are finding it to be more and more difficult to leave the house without feeling as if they are risking their lives. As such, for our project, we wanted to capture a bit of this feeling of uneasiness to illustrate to future generations what it was like to live during these times. We will do this mainly through the use of style transfer techniques, which will take the style of images of the coronavirus and apply them to images of things that one might see in his/her everyday life. We hope that people later on will be able to study and reference the narrative that we intend to create here in this project as a means of gaining insight into life during the COVID-19 pandemic.
  
As mentioned previously, to create these images, we intend to use the style transfer technique. We will be training our model using artistic representations of the coronavirus. Then, we will transfer the style of these images onto everyday objects and recognizable locations. To do this, we will be using a convolutional neural network similar to the one shown in class. By doing this, we hope to illustrate the gravity of our current situation as many people, even now, seem to be forgetting the consequences of ignoring social distancing measures.

Some other concerns that we have is that our style transfer will not be able to create a meaningful or artistically pleasing image. Since the coronavirus artistic representations have been created by different artists, we are worried that when we train our model, our model will not be able to capture the style of the images that it is trained on. As such, we plan to experiment with several different images and image styles to help establish our narrative better.


References

https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee
This reference shows us the basics of how to style transfer using neural networks.

https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/
This reference above is a project of style transfer performed on several images and also a live webcam. We think that this project is very interesting and could be very influential for us as it could potentially help us learn how to apply style transfers onto a live video, through our own webcams.

https://nico-opendata.jp/publish/neural-style-synthesizer/neural-style-synthesizer-slides.pdf
Reference for multiple image style transfer and how to effectively average the styles of multiple images and apply that average to a single image.



## Data and Model

In choosing our model, we decided that using a CNN, or convolutional neural network, would be most appropriate for our task as we are attempting to apply style transfer to a set of images, and CNNs are most commonly associated with analyzing visual imagery. For the most part, we followed the structure of the style_transfer_keras.ipynb notebook, which demonstrated style transfer using tensorflow. 

“A Neural Algorithm of Artistic Style”:  https://arxiv.org/pdf/1508.06576v2.pdf


For our images, we first needed to choose a set of base images that would serve as the main content for our project. Since our goal was to represent the effect of this virus on our lives, we purposely selected many public locations that we believed to be the most heavily impacted. In fact, many of the places that we chose to serve as examples of these times have either been completely shut down or are currently limited to a certain number of people as a result of the pandemic. We ultimately chose the following 8 locations/images:
 
- A playground: https://c1.peakpx.com/wallpaper/458/377/246/playground-slide-park-childhood-wallpaper-preview.jpg
- A basketball stadium: https://thenypost.files.wordpress.com/2020/03/march-madness-2020-no-brackets-ncaa.jpg?quality=80&strip=all&w=978&h=652
- A testing center: https://www.lalenguacaribe.co/wp-content/uploads/2020/06/1140-walmart-testing-site-esp.jpg
- A casino: https://ewscripps.brightspotcdn.com/dims4/default/7ed0b88/2147483647/strip/true/crop/1848x970+0+112/resize/1200x630!/quality/90/?url=https%3A%2F%2Fewscripps.brightspotcdn.com%2F23%2F03%2F0ca13ba64bdcb9d89e07781e7326%2Fscreen-shot-2020-02-19-at-6.53.13%20PM.png 
- A grocery store: https://www.galesburg.com/storyimage/LG/20200316/NEWS/200319835/EP/1/1/EP-200319835.jpg 
- The beach: https://media2.s-nbcnews.com/i/newscms/2020_18/3328241/200428-packed-beach-los-angeles-times-ew-530p_c7d3b1497fd65bd256dbe964dcf5fbf6.jpg 
- UCSD: https://media-exp1.licdn.com/media-proxy/ext?w=800&h=800&hash=UTMLCo1e5dEMRqk%2FmbpJMCPQ9pw%3D&ora=1%2CaFBCTXdkRmpGL2lvQUFBPQ%2CxAVta5g-0R6_kAgezBk28-CUrly0qkJDUM3SB3fiNGDqporRNDT2K56IFrCoo1gXeSsFmAQ6EOesSTfoR5C0eYOAN8Iry8A
- The Earth:
https://www.fr.de/bilder/2020/03/26/13629149/1757522012-feu_erde_2_270320-MpR4Vl6Y6ec.jpg

For our style images, we chose several different artistic representations of the COVID-19 virus and of antibodies. We wanted our style images to capture all sorts of different shapes and textures from a variety of representations that we found. This would allow us to visualize numerous effects on the final images and to generate a plethora of unique stylized photos of known and recognizable locations. As such, we ultimately chose the following 4 images:

- ‘cv’:
https://images.unsplash.com/photo-1584118624012-df056829fbd0?ixlib=rb-1.2.1
- ‘virus’: https://www.metrowestdailynews.com/storyimage/WL/20200326/NEWS/200328104/AR/0/AR-200328104.jpg
- ‘virus2’: https://img1.wsimg.com/isteam/ip/e9ad0ed6-9c30-45df-9f36-1877fa6e7716/hd%20image.jpg
- ‘antibodies’: https://foolde-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/GettyImages-516629150.jpg

## Code

Our code section consists of our main python script ‘transfer.py’ and an ‘images’ folder. ‘Transfer.py’ is where we execute the style transfer for all the images using a CNN. The images folder contains three separate folders within it: “bases,” “styles,” and “results.” The ‘bases’ folder holds all the main content images such as pictures of the beach, a stadium, etc. The ‘style’ folder holds all of the style images used for the transfer such as pictures of the coronavirus. And lastly, the ‘results’ folder is where our resulting style-transferred images are stored. Our python script will run through every image in the base folder and generate a unique style-transferred result of that image with every style in the ‘style’ folder. 

The script took a lot of influence from style_transfer_keras.ipynb, which is an implementation of neural style written with TensorFlow. The script initially grabs the base and style images, rescales them and then runs them through VGG19. Subsequently, it then gets the tensor representations of the images. From this, multiple loss functions are then calculated, and during each iteration, the loss is minimized to produce the style-transferred image. Each optimization iteration takes roughly 10 seconds on datahub; since we are running 20 iterations for each image, a style transfer on a single image should take around 3 minutes. Once the style transfer is complete, the resulting image can be seen in the ‘results’ folder with a name of the format: ‘name of base image_name of style image’.

## Results

In trying to obtain our desired result, we tried inputting different numbers of iterations for optimization. We noticed that as we increased the number of iterations, the input image that we are trying to transfer a specific style onto takes more of the form of the COVID-19 images’ attributes. We took this into account during our experimentation as we wanted it to be overtly obvious that our pictures were COVID-19 related. However, at the same time, we also wanted our resulting images to be recognizable in terms of the locations that they originally captured. Thus, we did not want to increase the number of iterations too much. We found that 20 iterations was a happy medium for most of our images. Here are some of the interesting results we found using different styles.

**Original**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/bases/beach.jpeg" width="40%" height="40%" title="beach">

**Style**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/styles/cv.jpg" width="40%" height="40%" title="cv">

**Result**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/results/result_beach_cv.png" width="40%" height="40%" title="beach_cv">

One of the base images that we decided to use was that of a crowded beach, something that used to be a common sight before the coronavirus pandemic. We performed style transfer on this image, using a different image of a rendering of the actual COVID-19 virus. The resulting image can be seen above. In the generated image, we can see that the style transfer seems to have replaced all of the people that were captured in the original image with black silhouettes outlined in red, similar to what thermal imaging might show. Some individuals from the base photo also now appear to be translucent in the generated image, which gives them a ghost-like appearance. The people highlighted in red also establish a violent and threatening mood. In general, applying the style transfer onto the image of the beach definitely gives it a more ominous, dangerous feel. 

**Original**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/bases/casino.png" width="40%" height="40%" title="casino">

**Style**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/styles/virus2.jpg" width="40%" height="40%" title="virus2">

**Result**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/results/result_casino_virus2.png" width="40%" height="40%" title="casino_virus2">

Another one of the base images that we used was this picture of a relatively uncrowded casino, taken somewhat recently to serve as an example of some businesses deciding to reopen, perhaps prematurely. We performed a style transfer on the image, using a rendering of what the COVID-19 virus may look like in the bloodstream. The resulting image, displayed above, was interesting in that the tone and feel of the image was completely different from that of the many of the other style transferred images, including the first one that we discussed above. Rather than being off-putting or “dangerous”, this style-transferred image gave off a more “cartoony” feel. One could almost imagine the image created here being displayed in an art gallery or perhaps appearing in a scene from Osmosis Jones. In general, this style transfer definitely gave off a more artistic vibe more so than did most of the other style transfers that were generated in this project. 

**Original**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/bases/testing.jpg" width="40%" height="40%" title="testing">

**Style**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/styles/antibodies.jpg" width="40%" height="40%" title="antibodies">

**Result**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/results/result_testing_antibodies.png" width="40%" height="40%" title="testing_antibodies">

The base image for this style transfer is one that depicts the distribution of virus testing kits using an image of antibodies as the style template. In the original image, the amount of protective gear being worn by the health official represents the need for safety, but consequently it also corresponds to the presence of danger. On the other hand, the image of the antibodies’ blue color attempts to shed a more positive outlook or feel to the image, as antibodies generally tend to be beneficial, and we can see that this is so in the generated image. We can also see that because the antibodies consist of smaller spherical particles that are bonded together, there are many objects in the resulting image that look as if they consist of many small circular components. This is best exemplified through the health official’s face shield as it appears to be entirely made of such particles. And lastly, we can also see that the style transfer process has much difficulty in dealing with the sky as it attempts to texture the shadows of the sky in the background.

**Original**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/bases/earth.jpg" width="40%" height="40%" title="earth">

**Style**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/data/styles/virus.jpg" width="40%" height="40%" title="virus">

**Result**

<img src="https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group6/blob/master/results/result_earth_virus.png" width="40%" height="40%" title="earth_virus">

The base image for this style transfer is a picture of the Earth which uses a much smoother representation of the coronavirus as its style template. As you may know, the Earth is something that is familiar to everyone simply because it is our home. It is the only place that we know is able to sustain life, and because life exists here, viruses exist as well to infect it. As such, looking at the resulting image, we can see that it is essentially covered in splotches, showing that the neural network heavily favored the protein spikes of the virus during the style transfer. What is also interesting is that the splotches were applied to the image in such a way that did not construct the Earth with the anatomy of the virus nor did it make the Earth look like the virus. Instead, the resulting image is much more reminiscent of microbes under a microscope. The splotches that are present in the entire image has the effect of making the Earth look contaminated, which is exactly how people perceive it now during this time.

## Discussion

Our plan to show how rampant and pervasive COVID-19 currently is proved to be successful. By using style transfer, we are able to transform any image’s style and form to that of our COVID-19 virus images. As intended, the resulting images give the viewer different impressions of how COVID-19 has taken control of the world similar to how the style of the COVID-19 images has taken over the base images. The resulting images also signify how these familiar places no longer look or function as they regularly did before the coronavirus pandemic.  Overall, we attempted to exemplify all of the feelings commonly expressed during the COVID-19 outbreak through our resulting style transfer images.

Compared to more traditional artwork, our approach was unique in that we did not have to manually draw up anything or have anything specific planned or thought up. While we did have a message that we intended to convey, our thought process only required thinking of which two images or ideas could go together in order to evoke that notion within the viewer. For our examples, we took locations that we thought our audience may know and combined them with artistic representations of the COVID-19 virus because we thought that taking someplace familiar and giving it a virus-like representation would best imply the invisible threat. We did not know exactly what our end product would look like, which contrasts with traditional approaches to doing artwork where the artist generally knows what to expect. As such, all of our resulting images are a product of trial and error. In fact, we had to try different parameters and inputs many times in order to find the aesthetic that we wanted. 

COVID-19 has transformed the lives of all people around the globe. By transferring the styles from the artistic renderings of the COVID-19 virus onto places that we are familiar with, we hope that our audience, or anyone who has been affected by the COVID-19 pandemic, in general, is able to relate to these images that we have generated. Seeing these familiar places overlaid with the style and form of the coronavirus, we wanted to remind people of some of the things that the COVID-19 virus has taken from us. Showing locations, such as UCSD and the beach, in the style of the COVID-19 virus renderings also serves to illustrate, in an exaggerated way, how the presence of COVID-19 has caused various locations to be physically altered. 

Like most, if not all, generative art, there are a range of ethical concerns associated with style transfer. For one, art generated by this method, by design, requires one to draw from other art styles, so the generated art isn’t completely “unique”. Furthermore, this type of art has limited human input. Art is usually defined as being the manifestation of human creativity or imagination. Some may argue that the “art” being generated here is only partially, if at all, fitting of the common definition of art. Essentially, and this criticism isn’t limited to only this form of generative art, art generated through style transfer blurs the line between what may be considered and what may not. Some of the other usual criticisms that are aimed at generative art, mainly those concerning the ability to fake something, probably don’t apply here, as it’s pretty obvious to see that these images are doctored, but this may change in the future, and may be a point of consideration for the creators and users of this tool. 

For the future, we would like to implement a real-time style transfer tool. We would like users to be able to apply style transfer directly onto what their webcam is seeing, so they too can create COVID-19 related artwork. Also, they would not have to go through the trial and error process like we did as they would be able to immediately see the result of the style transfer. Hopefully, this creates yet another way for us to show people how we felt during these precarious times.


## Team Roles

Iman Nematollahi: Data, Code, part of Discussion
Justin Lee: Part of Discussion, Result, and Slides
Soon Gi Shin: Abstract, Proofreading, part of Results, part of Discussion
Jaskaranpal Singh: Discussion, Result, and Slides
Dan Ngo: Discussion, Results, Slides

## Technical Notes and Dependencies

See imports in .py file
Unzipped Images folder and .py file must be in the same directory in order to run properly.

## Reference

style transfer notebook: https://github.com/roberttwomey/dsc160-code/blob/master/examples/style_transfer_tensorflow/style_transfer_keras.ipynb

https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee

https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/

https://nico-opendata.jp/publish/neural-style-synthesizer/neural-style-synthesizer-slides.pdf

