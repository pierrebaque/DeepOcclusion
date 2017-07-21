### Implementation of the Deep-Occlusion Reasoning
Paper: https://arxiv.org/abs/1704.05775 
Project page: https://pierrebaque.github.io/page-DeepOcclusion/ 

#### Dependencies
The main dependency is the Theano package.   
You also need to install the theano ROI-Pooling Layer from https://github.com/ddtm/theano-roi-pooling to get the unary potentials to work.

#### How to run.
*Nb: The demo provided here is temporary and should be improved soon to make it more robust and easier to run. Happy to read your feedbacks.*

Clone the github code

Download this zip file https://drive.google.com/file/d/0BxijKKbgC7wRTE03ajEtZ1FYc2M/view?usp=sharing

Unzip it in the root directory

From the same root directory, type:

mv tozip/models VGG/models  
mkdir Potentials/Parts/Run  
mkdir Potentials/Unaries/Run   

Then, run the notebooks in the following order:   
RunParts.ipynb   
RunUnaries.ipynb   
RunPom.ipynb   


You should get some results. This is for running the pretrained model and not for training yet. 
