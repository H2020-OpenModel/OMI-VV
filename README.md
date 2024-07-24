# V&V services for OpenModel
 Updated to work on OMI

## Installation 

Install directly from GitHub

`pip install git+https://github.com/H2020-OpenModel/OMI-VV.git`

## Usage 

See examples directory for examples of use

Step 1:

After installation the vv services can be imported as a normal Python package. 

`from vv.vv import verification`

An interactive widget is loaded for the user. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/images/image1.png)


After the widget is loaded the user can fill out the required fields. Firstly, the database/ knowledge graph to train from. In this case `https://openmodel.app/fuseki3/dataset`. 
Secondly, the desired prediction values. Finally, the keys for training are added. In this case, we're training from atomisation energy and distance values in the knowledgebase, where we're asking the ML 
model to predict the atomisation energy at unknown distance values, 1.1, 1.2, and 1.3 Ang. We fill the plot box to get a visual output, and run the V&V.

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/images/image2.png)


The `verification()` function uses a neural network (multi layer process regression) to predict new values, in this case atomisation energy at distance 1.1, 1.2, 1.3 Ang.
Output.txt will contain the predicted values, and a confidence interval which is determined through conformal prediction. If plotting is specified, a plot of the predicted values (output.png) is also created. 


![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/images/image3.png)