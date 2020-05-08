## Machine learning project evolution

The goal of this project is to show my vision of how you could manage a complexity of 

### Intro
I love jupyter notebooks. They are very useful especially when you want to do a fast check of your idea. It is convenient to show
examples and do a visualation.
However jupyter have some disadvantages that you should consider. For example it is very easy to start writing bad code like do a code duplication or put all your code in one file. Another downside is lack of code tracking. When you commit your code it's hard to visually understand what has changed without 3rd party tools.

![Image of git diff](https://blogamirathi.files.wordpress.com/2018/07/screen-shot-2018-07-23-at-4-51-38-pm1.png?w=500)

[Image ref:](https://blog.amirathi.com/2018/07/23/github-with-jupyter-notebook/)

### Why i made it
By this project i wanted to show how you could apply very basic refactoring techniques to make your code better.

### More context
I want to provide a more context and say that this code was supposed to be run on colab.google and kaggle. This imposes restrictions on what the executable file should be. For example you can't launch .py file on google colab. For this you should use .ipynb notebooks.

### How to use it
**stage_01_initial_experiment**  contains a single notebook. There you can find a dirty code that i used to check an article about segmenation in tf https://www.tensorflow.org/tutorials/images/segmentation 
It perfectly fine to start with code like this and check if everything works

**stage_02_more_notebooks**  after a few iterations i needed a script for visualization. The new script (Show results.ipynb) uses the same model as in Train.ipynb, however i just duplicated DataLoader and model code. Every time when you have a code duplication it is a sign that you should do a refactoring. 

**stage_03_extract_code** to make a code better i extract DataLoader and the model to separate files

**stage_04_make_code_trackable/segmentation** At this stage i expect that i won't have big changes in a training process and i want to make a training launcher more flexible and more friendly for git. For this i do the next thing
1. Move all the from a train notebook to the .py file
2. Add a configuration object that i pass to the SegmentationNetwork class.
3. Create another ipynb file which i use only to prepare the configuration and start training.

### Dataset
For the project i've used a public dataset provided by the great tool http://supervise.ly/
https://hackernoon.com/releasing-supervisely-person-dataset-for-teaching-machines-to-segment-humans-1f1fc1f28469

For the conveniece of the reader i could propose to use my link
https://www.dropbox.com/s/8zsjwoirx53eltg/supervisely_converted.zip?dl=1

There you can find the same images as in the original dataset, but all masks are converted to png files.
