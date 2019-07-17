
(
var synth1_pitch = Bus.control(s);
var synth1_vol = Bus.control(s);
var synth1_color = Bus.control(s);
var synth2_pitch = Bus.control(s);
var synth2_vol = Bus.control(s);

var mixer_synth1_vol = Bus.control(s);
var mixer_synth2_vol = Bus.control(s);

var bus_synth1 = Bus.audio(s);
var bus_synth2 = Bus.audio(s);

var global_lag = 3;
var global_decay = 2;

var oscOutForTraining = NetAddr.new("2.0.0.1", 8005);
var trainingsData = [0.0, 0.0, 0.0, 0.0, 0.0];

///////////////////////////////////////////////////////////
//   OSC
///////////////////////////////////////////////////////////
OSCdef("synth1/pitch",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth1_pitch.set(msg[1]);
		trainingsData[0] = msg[1];
		oscOutForTraining.sendMsg("/record_sound_vector", trainingsData[0], trainingsData[1], trainingsData[2], trainingsData[3], trainingsData[4]);
		trainingsData.postln;
	}, "synth1/pitch"
);

OSCdef("synth1/vol",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth1_vol.set(msg[1]);
		trainingsData[1] = msg[1];
	}, "synth1/vol"
);

OSCdef("synth1/color",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth1_color.set(msg[1]);
		trainingsData[2] = msg[1];
	}, "synth1/color"
);

OSCdef("synth2/vol",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth2_vol.set(msg[1]);
		trainingsData[3] = msg[1];
	}, "synth2/vol"
);

OSCdef("synth2/pitch",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth2_pitch.set(msg[1]);
		trainingsData[4] = msg[1];
	}, "synth2/pitch"
);

OSCdef("mixer/synth1/vol",
	{
		arg msg;
		//msg.postln;
		mixer_synth1_vol.set(msg[1]);
	}, "mixer/synth1/vol"
);

OSCdef("mixer/synth2/vol",
	{
		arg msg;
		//msg.postln;
		mixer_synth2_vol.set(msg[1]);
	}, "mixer/synth2/vol"
);

OSCdef("sound",
	{
		arg msg;
		msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		synth1_pitch.set(msg[1]);
		synth1_vol.set(msg[2]);
		synth1_color.set(msg[3]);
		synth2_vol.set(msg[4]);
		synth2_pitch.set(msg[5]);
	}, "sound"
);

///////////////////////////////////////////////////////////
//   Mixer
///////////////////////////////////////////////////////////
EnvGen
(a = {
	arg t_env;
	var env = Env.new([1,1,0], [0.1,global_decay]).ar(0, t_env); //[0.5 -> hold time (max dur between osc msgs), 1 -> release time]
	var result = Mix.ar([In.ar(bus_synth1)*In.kr(synth1_vol).lag(global_lag)*In.kr(mixer_synth1_vol).lag(global_lag), In.ar(bus_synth2)*In.kr(synth2_vol).lag(global_lag)*In.kr(mixer_synth2_vol).lag(global_lag)])*env*4;
	result = FreeVerb.ar(result);
	Splay.ar(result);
}.play;
);



///////////////////////////////////////////////////////////
//   Synths
///////////////////////////////////////////////////////////
(

{
	var spec1 = [30, 3000, 'lin'].asSpec;
	var freq = spec1.map(In.kr(synth1_pitch).lag(global_lag))/ SampleRate.ir;
	var distance = 3.00;
	var spec2 = [0.42, 0.9, 'lin'].asSpec;
	var index = spec2.map(In.kr(synth1_color).lag(global_lag)); //MouseY.kr(0.42, 0.99);
	var theta, beta, num, denom, son;


	// Two phasors which will ramp from zero to 2pi
	theta = Phasor.ar(0, freq, 0, 2pi);
	beta  = Phasor.ar(0, freq * distance, 0, 2pi);

	num = sin(theta) - (index * sin(theta - beta));
	denom = 1 + index.squared - (2 * index * cos(beta));

	son = num / denom;
	son = son * 0.004 * (10/(index+0.5));

	Out.ar(bus_synth1, son);

}.play;  // Use ".freqscope" or ".scope", both are illustrative.
);

(
{
	var freq = 40 * (In.kr(synth2_pitch).lag(global_lag) * 103).midiratio;
	var osc = WhiteNoise.ar(); //freq: freq, mul: In.kr(synth1_vol).lag(0.5));
	osc = BBandPass.ar(in: osc, freq: freq, bw: 0.2, mul: 0.6 );
	Out.ar(bus_synth2, osc);
}.play
)

)










//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


a = {arg osc, argu = "me"; osc = 0; (osc+argu+"hello").postln};
a.value("lkjdlsfk")



(
~padSynth = {
	arg ctrl1 = #[0];

	var osc;
	//osc = LFSaw.ar(100 * ([0,7,12,15]-12).midiratio, mul:1) * Env.perc(0,0.3).ar(0,Impulse.ar(10.collect{arg i; i.lincurve(0.0,10.0,0.2,0.8, -2)}));
	//osc = LFSaw.ar(100 * ([0,7,12,15]-12 + (ctrl1*10)).midiratio, mul:0.5);
	osc = SinOsc.ar(10000 * ctrl1+30, mul:0.2);
	osc = HPF.ar(osc, freq: 80, mul: 1.0, add:0.0);
	//osc = LPF.ar(osc,[3000,400, 10000, 2000] );
	//osc = HPF.ar(osc,[400,100, 1000]) * (-20).dbamp;
	//osc = osc + DelayN.ar(osc,0.5, [0.1,0.3]).scope;

	//osc = SinOsc.ar(freq = 440, phase =0.0, mul = 0.5, add = 0);
	Splay.ar(osc);
}.play;

)




