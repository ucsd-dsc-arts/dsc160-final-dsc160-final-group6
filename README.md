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
- A casino:
https://ewscripps.brightspotcdn.com/dims4/default/7ed0b88/2147483647/strip/true/crop/1848x970+0+112/resize/1200x630!/quality/90/?url=https%3A%2F%2Fewscripps.brightspotcdn.com%2F23%2F03%2F0ca13ba64bdcb9d89e07781e7326%2Fscreen-shot-2020-02-19-at-6.53.13%20PM.png 
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

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for: 

- code for data acquisition/scraping
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

## Results

(30 points) 

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below: 

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

## Discussion

(30 points, three to five paragraphs)

The first paragraph should be a short summary describing your results.

The subsequent paragraphs could address questions including:
- Why is this culturally innovative?
- How does your generative computational approach differ from traditional art/music/cultural production? 
- How do your results relate to broader social, cultural, economic political, etc., issues? 
- What are the ethical concerns for this form of generative art? 
- In what future directions could you expand this work?

## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.

## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
- Blog posts
