# V&V services for OpenModel
 Updated to work on OMI


## Introduction 

In the Verification and Validations services, verification is defined as confirmation through objective evidence to 
determine whether specified requirements have been fulfilled, where the objective evidence for verification may be determined 
through performing alternative and additional calculations.  
For OpenModel, verification addresses such user question as “are we implementing the model correctly?”.  
Validation is defined as confirmation through the provision of objective evidence which ensure the requirements for specified application have been fulfilled, 
which may correspond to performing additional calculations. 
In that respect, validation addresses the more overarching question of “are the results correct?” which are important concepts 
when attempting to take well-informed decisions for the design of new materials based on simulations. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure1.png)

## Installation 

Install directly from GitHub

`pip install git+https://github.com/H2020-OpenModel/OMI-VV.git`

## Examples 

Full examples of using the V&V are found in the examples folder. 


### Using the V&V with ExecFlow 


In this section, an example of running the V&V with ExecFlow is demonstrated. ExecFlow executes workflows specified by previous 
steps in the OpenModel Open Innovation Platform (OIP), such as OntoFlow. Herein an example of a density functional theory (DFT) quantum chemical simulation 
performed by ExecFlow and Quantum Espresso code on face-centred cubic (FCC) silicon is demonstrated, with subsequent verification and validation steps. 

Below is an example of a YAML file specifying an ExecFlow workflow for executing a DFT calculation on silicon. Under ‘data’ the calculation 
parameters for Quantum Espresso are specified, with the type of calculation, pseudopotential, numerical parameters and material structure included. 
Once the input data is specified the steps in the workflow are defined under the “steps” key.  For this example, a numerical parameter of the calculation 
called “kpoints” needs to be validated by the V&V services. A k-point setting is an important parameter to converge when performing DFT calculation on periodic systems,
such as FCC silicon, as this will affect the accuracy of the outputted energy. In this example, it is important to check that the k-point parameter chosen is converged 
with respect to the output energy. ExecFlow takes in postprocessing steps with the “postprocess” key, where the V&V services is called.  
Firstly, the user defines the knowledgebase for retrieving data for the V&V services through the “database” key, and in this case, the calculation 
connects to the default shared OpenModel knowledgebase, ‘https://openmodel.app/kb/data’. Secondly, the user defines triples data to parse into the 
V&V services with “Key1” and “Key2”. The V&V services work with subjects, predicates, and object as keys for retrieving training data, in this case the 
predicates of the values stored in the desired knowledge graph are used “has_kpoint_mesh” and “has_total_energy”. Finally, the user defined the numerical parameter 
needing to be validated under the “prediction” key, which is ‘kpoints’ in this example. This is achieved through calling on the variable 
#/data/kpoints_scf which was defined in the previous step of the workflow. Once the calculation and V&V workflow is defined, the YAML is executed as normal with ExecFlow. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure2.png)

Once the workflow is executed by ExecFlow, a Quantum Espresso calculation is performed on the silicon model, depicting below
the workflow inputs and output of the calculation. When the calculation has completed, the postprocessing steps are executed, in this case the V&V. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure3.png)

The chosen k-point setting is parsed to the MLP-NN model of the V&V services, where the model will make a prediction on this value based on training 
from similar data stored in the knowledgebase, which were specified by “Key1” and “Key2”. The MLP-NN model will output a prediction 
for energy at this k-point setting, with an upper energy value and lower energy value. In this case, the V&V services produce an upper and lower energy 
value of -148.33 and -153.60 eV, meaning that the user should expect energy value of the calculation to be within this limit. Here, the Quantum Espresso 
calculation with k-point setting of 4x4x4 outputted an energy value of -152.40 eV, which are within the limits predicted by the V&V service, therefore validating 
the results of the calculation. Finally, the results from the V&V are ontologised (with a .ttl file produced), depicting blow is the input parameter for validation (k-point), 
the predictions made by the MLP-NN model, and the provenance of the data, showing the parent graph in the knowledgebase from where the training data was 
obtained. The V&V results are subsequently uploaded back into the knowledgebase, with the validation results of the specified workflow stored for future reference. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure4.png)

### Using the GUI

Widgets in JupyterHub are supported so the user can interactively use the V&V service through tne GUI.

Step 1:

After installation the vv services can be imported as a normal Python package. 

`from vv.vv import verification`

An interactive widget is loaded for the user. 

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure5.png)


After the widget is loaded the user can fill out the required fields. Firstly, the database/ knowledge graph to train from. In this case `https://openmodel.app/fuseki3/dataset`. 
Secondly, the desired prediction values. Finally, the keys for training are added. In this case, we're training from atomisation energy and distance values in the knowledgebase, where we're asking the ML 
model to predict the atomisation energy at unknown distance values, 1.1, 1.2, and 1.3 Ang. We fill the plot box to get a visual output, and run the V&V.

![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/Images/Figure6.png)


The `verification()` function uses a neural network (multi layer perception regression) to predict new values, in this case atomisation energy at distance 1.1, 1.2, 1.3 Ang.
Output.txt will contain the predicted values, and a confidence interval which is determined through conformal prediction. If plotting is specified, a plot of the predicted values (output.png) is also created. 


![alt text](https://github.com/H2020-OpenModel/OMI-VV/blob/main/images/Figure7.png)
