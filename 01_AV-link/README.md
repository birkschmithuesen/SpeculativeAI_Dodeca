# SpeculativeArtificialIntelligence / Exp. #1 (audiovisual associations)
The program generates a model of feed-forward NN with 30 inputs and 13824 outputs.
30 inputs: FFT analyses of sounds
13824 outputs: sets the brightness for 13824 LEDs of the lightobject "Interspace #3"

The training data can be downloaded here:
www.birkschmithuesen.cpom/SAI/traingsdata.txt

The program predicts the output for the lightobject in real time from received FFT data.
The communication is done via OSC.
##WARNING: if you have no network card with the fixed IP 2.0.0.1 in your computer, the program will crash. Also see: line 153 and line 102

Here is a basic diagram how the communication between the program used is working right now:

Ableton Live(sound program)/FFT analysis => 30 float values via OSC (NN input) => python/neural network => 13824 float values via OSC (NN output) => JAVA(visualizer on screen and light object) 
