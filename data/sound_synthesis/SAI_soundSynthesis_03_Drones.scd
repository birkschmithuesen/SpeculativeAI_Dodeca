
(




var bassDrone_vol = Bus.control(s, 1);
var bassDrone_bw = Bus.control(s, 1);
var bassDrone_freq = Bus.control(s, 1);
var midDrone_vol = Bus.control(s, 1);
var midDrone_bw = Bus.control(s, 1);
var midDrone_freq = Bus.control(s, 1);
var midDrone_mix = Bus.control(s, 1);
var lead_vol = Bus.control(s, 1);
var lead_pitch = Bus.control(s, 1);

var bus_bassDrone = Bus.audio(s, 2);
var bus_midDrone = Bus.audio(s, 2);
var bus_lead = Bus.audio(s, 2);

var mixer_bassDrone = 1;
var mixer_midDrone = 1;
var mixer_lead = 1;
var initializeSynth = 100;

var global_lag = 3;
var global_decay = 2;

var oscOutForTraining = NetAddr.new("2.0.0.1", 8005);
var trainingsData = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

///////////////////////////////////////////////////////////
//   OSC
///////////////////////////////////////////////////////////

OSCdef("sendTrainData",
	{
		arg msg;
		oscOutForTraining.sendMsg("/record_sound_vector", trainingsData[0], trainingsData[1], trainingsData[2], trainingsData[3], trainingsData[4], trainingsData[5], trainingsData[6]);
		trainingsData.postln;
	}, "sendTrainData"
);

OSCdef("bassDrone/vol",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		bassDrone_vol.set(msg[1] * mixer_bassDrone);
		trainingsData[0] = msg[1];
	}, "bassDrone/vol"
);

OSCdef("bassDrone/bw",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		bassDrone_bw.set(msg[1]);
		trainingsData[1] = msg[1];
	}, "bassDrone/bw"
);

OSCdef("bassDrone/freq",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		bassDrone_freq.set(msg[1]);
		trainingsData[2] = msg[1];
	}, "bassDrone/freq"
);

OSCdef("midDrone/vol",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		midDrone_vol.set(msg[1] * mixer_midDrone);
		trainingsData[3] = msg[1];
		//Here comes a very stupid workaround to set the twop Oscilators to different frequencies. I don't know how to do it within in the initialization process.
		if (initializeSynth > 0){
			initializeSynth = initializeSynth - 1;
		};
		if (initializeSynth == 1) {
		(
			~synths_mid[0].set(\freq, 440);
			~synths_mid[1].set(\freq, 655, \amp, 0.2);
		)
		};

	}, "midDrone/vol"
);

OSCdef("midDrone/bw",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		midDrone_bw.set(msg[1]);
		trainingsData[4] = msg[1];
	}, "midDrone/bw"
);

OSCdef("midDrone/freq",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		midDrone_freq.set(msg[1]);
		trainingsData[5] = msg[1];
	}, "midDrone/freq"
);

OSCdef("lead/vol",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		lead_vol.set(msg[1] * mixer_lead);
		trainingsData[6] = msg[1];
	}, "lead/vol"
);

OSCdef("lead/pitch",
	{
		arg msg;
		//msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		lead_pitch.set(msg[1]);
		trainingsData[7] = msg[1];
	}, "lead/pitch"
);

OSCdef("mixer/bassDrone",
	{
		arg msg;
		//msg.postln;
		mixer_bassDrone = msg[1];
	}, "mixer/bassDrone"
);

OSCdef("mixer/midDrone",
	{
		arg msg;
		//msg.postln;
		mixer_midDrone = msg[1];
	}, "mixer/midDrone"
);

OSCdef("mixer/lead",
	{
		arg msg;
		//msg.postln;
		mixer_lead = msg[1];
	}, "mixer/lead"
);

OSCdef("sound",
	{
		arg msg;
		msg.postln;
		a.set(\t_env, 1); //call this everytime an osc msg comes in
		bassDrone_vol.set(msg[1]);
		bassDrone_bw.set(msg[2]);
		bassDrone_freq.set(msg[3]);
		midDrone_vol.set(msg[4]);
		midDrone_bw.set(msg[5]);
		midDrone_freq.set(msg[6]);
		lead_vol.set(msg[7]);
		lead_pitch.set(msg[8]);
	}, "sound"
);

///////////////////////////////////////////////////////////
//   Mixer
///////////////////////////////////////////////////////////
EnvGen
(a = {
	arg t_env;
	var env = Env.new([1,1,0], [0.1,global_decay]).ar(0, t_env); //[0.5 -> hold time (max dur between osc msgs), 1 -> release time]
			var result = Mix.ar([In.ar(bus_bassDrone)*In.kr(bassDrone_vol).lag(global_lag), In.ar(bus_midDrone)*In.kr(midDrone_vol).lag(global_lag), In.ar(bus_lead)*In.kr(lead_vol).lag(global_lag)])*env;
	result = FreeVerb.ar(result);
	Splay.ar(result);
}.play;
);



///////////////////////////////////////////////////////////
//   Synth - bassDrone - inspired by http://sccode.org/1-4V1
///////////////////////////////////////////////////////////

(

/*
s.freeAll;
Window.closeAll;
Buffer.freeAll;
*/

/*
Function for interpolating between two
values based on an interpolation value:

0 ==> old, 1 ==> new,
0.5 ==> halfway between old/new, etc
*/
~interpFn = {
	arg old=20, new=10, bal=0.5;
	old + ((new - old) * bal);
};

/*
three wacky envelopes, each converted to Signal format.
each has 12 level points. the first and last value are
always zero. the inner 10 points are random between -1
and +1. the internal functions normalize the levels so
the highest value is always +/-1.0
*/
~wt0 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

~wt1 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

~wt2 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

/*
Signals that contain values representing interpolations between two wavetables.
~i0 interpolates between ~wt0 <==> ~wt1
~i1 interpolates between ~wt1 <==> ~wt2
*/
~i0 = ~wt0.copy;
~i1 = ~wt2.copy;

s.waitForBoot({

	//load signals to buffers in wavetable format
	~wt0Buf = Buffer.loadCollection(s, ~wt0.asWavetable);
	~wt1Buf = Buffer.loadCollection(s, ~wt1.asWavetable);
	~wt2Buf = Buffer.loadCollection(s, ~wt2.asWavetable);
	~i0Buf = Buffer.alloc(s, 1024, 1);
	~i1Buf = Buffer.alloc(s, 1024, 1);
	s.sync;
	~i0Buf.setn(0, ~i0.asWavetable);
	~i1Buf.setn(0, ~i1.asWavetable);

	s.sync;

	/*
	Create two Osc Synths, reading their wavetable
	data from ~i0Buf and ~i1Buf
	*/
	~synths = [~i0Buf, ~i1Buf].collect{
		arg buf, i;
		{
			/*
			detune=0 makes a much more boring sound, but also
			makes the wavetable interpolation more
			observable (evaluate s.scope and adjust the
			horizontal slider to watch the waveform)
			*/
			arg detune=0.1, freq=50, amp=0.3;
			var sig;
			var spec1 = [40, 15000, 'exp'].asSpec;
			var bandPassFreq = spec1.map(In.kr(bassDrone_freq).lag(global_lag));
			var spec2 = [0.1, 5.0, 'lin'].asSpec;
			var bandPassBW = spec2.map(In.kr(bassDrone_bw).lag(global_lag));
			sig = Osc.ar(
				buf.bufnum,

				//frequency with random moving detune value (in semitones)
				freq * LFNoise1.kr({Rand(0.08,0.15)}!8).bipolar(detune).midiratio,

				{Rand(0,2pi)}!8
			);

			//spread 8-channel detuned Osc texture across two channels
			sig = Splay.ar(sig);

			//avoid funky DC bias
			sig = LeakDC.ar(sig);

			sig = sig * amp;

			//sig = BLowPass.ar(sig, MouseY.kr(40, 10000,\exponential));

			sig = BBandPass.ar(sig, bandPassFreq, bandPassBW, (8.0-bandPassBW)/2);

		}.play(outbus:bus_bassDrone, fadeTime:4);
	};

	{
		arg min=0.2, max=1;
		var sig;
		sig = LFDNoise1.kr(
			LFNoise1.kr(8!2).exprange(min,max)
		).unipolar(1);

		/*
		uncomment this line to control wavetable
		interpolation with horizontal mouse position
		instead of a noise generator
		*/
		sig = 0!2; //interpolate to 0

		SendReply.kr(Impulse.kr(40), '/mouse', sig);
		0
	}.play(target:h, outbus:bus_bassDrone);

	s.sync;


	OSCdef(\mouse, {
		arg msg;

		/*
		when SendReply sends a value to the language,
		use that value as an interpolation parameter
		and update the wavetables from which the Osc
		UGen is reading
		*/

		//modify values
		~i0.waveFill({
			arg x, val, i;
			~interpFn.(~wt0[i], ~wt1[i], msg[3])
		});
		~i1.waveFill({
			arg x, val, i;
			~interpFn.(~wt1[i], ~wt2[i], msg[4])
		});

		//dynamically update Buffers
		~i0Buf.setn(0, ~i0.asWavetable);
		~i1Buf.setn(0, ~i1.asWavetable);
	}, '/mouse').permanent_(true);
})
);


///////////////////////////////////////////////////////////
//   Synth - midDrone - inspired by http://sccode.org/1-4V1
///////////////////////////////////////////////////////////

(

/*
s.freeAll;
Window.closeAll;
Buffer.freeAll;
*/

/*
Function for interpolating between two
values based on an interpolation value:

0 ==> old, 1 ==> new,
0.5 ==> halfway between old/new, etc
*/
~interpFn = {
	arg old=20, new=10, bal=0.5;
	old + ((new - old) * bal);
};

/*
three wacky envelopes, each converted to Signal format.
each has 12 level points. the first and last value are
always zero. the inner 10 points are random between -1
and +1. the internal functions normalize the levels so
the highest value is always +/-1.0
*/
~wt0 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

~wt1 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

~wt2 = Env(
	[0]++
	{
		var levs, peak;
		levs = {rrand(-1.0,1.0)}!10;
		peak = levs.abs.maxItem;
		levs = levs * peak.reciprocal;
	}.value ++
	[0],
	{exprand(0.01,1)}!11,
	{exprand(0.1,4)}!11
).asSignal(512);

/*
Signals that contain values representing interpolations between two wavetables.
~i0 interpolates between ~wt0 <==> ~wt1
~i1 interpolates between ~wt1 <==> ~wt2
*/
~i0 = ~wt0.copy;
~i1 = ~wt2.copy;

s.waitForBoot({

	//load signals to buffers in wavetable format
	~wt0Buf = Buffer.loadCollection(s, ~wt0.asWavetable);
	~wt1Buf = Buffer.loadCollection(s, ~wt1.asWavetable);
	~wt2Buf = Buffer.loadCollection(s, ~wt2.asWavetable);
	~i0Buf = Buffer.alloc(s, 1024, 1);
	~i1Buf = Buffer.alloc(s, 1024, 1);
	s.sync;
	~i0Buf.setn(0, ~i0.asWavetable);
	~i1Buf.setn(0, ~i1.asWavetable);

	s.sync;

	/*
	Create two Osc Synths, reading their wavetable
	data from ~i0Buf and ~i1Buf
	*/
	~synths_mid = [~i0Buf, ~i1Buf].collect{
		arg buf, i;
		{
			/*
			detune=0 makes a much more boring sound, but also
			makes the wavetable interpolation more
			observable (evaluate s.scope and adjust the
			horizontal slider to watch the waveform)
			*/
			arg detune=0.22, freq=440, amp=0.2;
			var sig;
			var spec1 = [110, 15000, 'exp'].asSpec;
			var bandPassFreq = spec1.map(In.kr(midDrone_freq).lag(global_lag));
			var spec2 = [0.1, 3.0, 'lin'].asSpec;
			var bandPassBW = spec2.map(In.kr(midDrone_bw).lag(global_lag));
			sig = Osc.ar(
				buf.bufnum,

				//frequency with random moving detune value (in semitones)
				freq * LFNoise1.kr({Rand(0.08,0.15)}!8).bipolar(detune).midiratio,

				{Rand(0,2pi)}!8
			);

			//spread 8-channel detuned Osc texture across two channels
			sig = Splay.ar(sig);

			//avoid funky DC bias
			sig = LeakDC.ar(sig);

			sig = sig * amp;

			//sig = BLowPass.ar(sig, MouseY.kr(40, 10000,\exponential));

			sig = BBandPass.ar(sig, bandPassFreq, bandPassBW, (8.0-bandPassBW)/2);

		}.play(outbus:bus_midDrone, fadeTime:4);
	};

	{
		arg min=0.2, max=1;
		var sig;
		sig = LFDNoise1.kr(
			LFNoise1.kr(8!2).exprange(min,max)
		).unipolar(1);

		/*
		uncomment this line to control wavetable
		interpolation with horizontal mouse position
		instead of a noise generator
		*/
		sig = 0!2; //interpolate to 0

		SendReply.kr(Impulse.kr(40), '/mouse', sig);
		0
	}.play(target:h, outbus:bus_midDrone);

	s.sync;


	OSCdef(\mouse, {
		arg msg;

		/*
		when SendReply sends a value to the language,
		use that value as an interpolation parameter
		and update the wavetables from which the Osc
		UGen is reading
		*/

		//modify values
		~i0.waveFill({
			arg x, val, i;
			~interpFn.(~wt0[i], ~wt1[i], msg[3])
		});
		~i1.waveFill({
			arg x, val, i;
			~interpFn.(~wt1[i], ~wt2[i], msg[4])
		});

		//dynamically update Buffers
		~i0Buf.setn(0, ~i0.asWavetable);
		~i1Buf.setn(0, ~i1.asWavetable);
	}, '/mouse').permanent_(true);
})
);

(

{
	var spec1 = [1300, 16000, 'exp'].asSpec;
	var freq = spec1.map(In.kr(lead_pitch).lag(global_lag))/ SampleRate.ir;
	var distance = 3.00;
	var spec2 = [0.42, 0.9, 'lin'].asSpec;
	var index = 0.0; //spec2.map(In.kr(lead_pitch).lag(global_lag)); //MouseY.kr(0.42, 0.99);
	var theta, beta, num, denom, son;


	// Two phasors which will ramp from zero to 2pi
	theta = Phasor.ar(0, freq, 0, 2pi);
	beta  = Phasor.ar(0, freq * distance, 0, 2pi);

	num = sin(theta) - (index * sin(theta - beta));
	denom = 1 + index.squared - (2 * index * cos(beta));

	son = num / denom;
	son = son * 0.8;
	//son = Dis
	Out.ar(bus_lead, son);

}.play;  // Use ".freqscope" or ".scope", both are illustrative.
)


)



