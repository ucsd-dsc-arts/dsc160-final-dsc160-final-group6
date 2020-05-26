# Project Title

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
  
  For our reach goal, we will attempt to apply the coronavirus style transfer using a live visual input. If we are able to achieve this, we will be able to apply this style transfer to a live input, which might further emphasize to others the direness of our present situation. We found an article demonstrating how to do this, but we do not currently know how feasible it is for us to be able to apply this technique given our time constraint.
  
  Some other concerns that we have is that our style transfer will not be able to create a meaningful or artistically pleasing image. Since the coronavirus artistic representations have been created by different artists, we are worried that when we train our model, our model will not be able to capture the style of the images that it is trained on. As such, we plan to experiment with several different images and image styles to help establish our narrative better.

References

https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee

This reference shows us the basics of how to style transfer using neural networks

https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/

This reference above is a project of style transfer performed on several images and also a live webcam. I think this project is very interesting and could be very influential for us and it could potentially help us learn how to apply style transfers onto a live video, through our own webcams.

https://nico-opendata.jp/publish/neural-style-synthesizer/neural-style-synthesizer-slides.pdf

Reference for multiple image style transfer and how to effectively average the styles of multiple images and apply that average to a single image.



## Data and Model

(10 points) 

In the final submission, this section will describe both the data you use for this project and any pre-existing models/neural nets. For each you should provide the name, a textual description, and a link. If there is a paper (for neural net) link that as well.
- Such and such Neural Net. The short description of this neural net. 
  - [link to code]().
  - [Title of Paper with Link](). 
- Training data. Short description of training data including bibliographic info. [link to data]().

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
