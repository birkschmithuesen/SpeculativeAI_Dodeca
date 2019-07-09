a = {arg osc, argu = "me"; osc = 0; (osc+argu+"hello").postln};
a.value("lkjdlsfk")



(
~padSynth = {
	arg ctrl1 = #[0];

	var osc;
	//osc = LFSaw.ar(100 * ([0,7,12,15]-12).midiratio, mul:1) * Env.perc(0,0.3).ar(0,Impulse.ar(10.collect{arg i; i.lincurve(0.0,10.0,0.2,0.8, -2)}));
	//osc = LFSaw.ar(100 * ([0,7,12,15]-12 + (ctrl1*10)).midiratio, mul:0.5);
	osc = SinOsc.ar(10000 * ctrl1+30, mul:0.2);
	//osc = LPF.ar(osc,[3000,400, 10000, 2000] );
	//osc = HPF.ar(osc,[400,100, 1000]) * (-20).dbamp;
	//osc = osc + DelayN.ar(osc,0.5, [0.1,0.3]).scope;

	//osc = SinOsc.ar(freq = 440, phase =0.0, mul = 0.5, add = 0);
	Splay.ar(osc);
}.play;

)


(
OSCdef(
	"pitch"
	,
	{
		arg msg;
		msg.postln;

		~padSynth.set(\ctrl1,msg[1..]);

	}, "/sound"
);
)


a =[1,2,3,4,5]
a[1]

[1,2,3,4,5][3..]











)
s.boot
a = 9

~myvar = 3




Integer

34.even

0.midiratio

Env.perc().plot



~padSynth.set(\ctrl1,1)
