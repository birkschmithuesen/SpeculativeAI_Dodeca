# SpeculativeArtificialIntelligence
An aesthetic research series to audiovisually translate the behaviour of neural networks

## Exp. #1 (audiovisual associations)
A neural network is trained with sound input (from FFT analysis) and visual output (light body on 13824 LEDs in volumetric light object). Each LED is connected to one output neuron representing its state with its brightness. <br>
For more details and code see the folder <b>01_AV-link</b>. <br>
Issues for that project are tagged with the label <b>01_AV-link</b>.

## Exp. #2 (conversation)
Two independent neural networks are trained with sound->visual and visual->sound associations. They are connected to each other in a closed circuit over the analog world with a speaker->microphone and light object->camera. <br>
For more details and code see the folder <b>02_Conversation</b>. <br>
Issues for that project are tagged with the label <b>02_Conversation</b>.

## packages needed
TensorFlow (pip tensorflow)
Keras (pip keras)
PythonOsc (pip python-osc)
