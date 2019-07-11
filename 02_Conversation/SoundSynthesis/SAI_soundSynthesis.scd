
(
b = Bus.control(s, 1);
OSCdef("pitch",
	{
		arg msg;
		msg.postln;
		//~padSynth.set(\ctrl1,msg[1..]);
		a.set(\t_env, 1, \bus, b);//call this everytime an osc msg comes in
		b.set(msg[1])
	}, "/sound"
);



EnvGen
(a = {
	arg t_env, bus;
	var env;
	env = Env.new([1,1,0], [0.5,1.3
	]).ar(0, t_env); //[0.5 -> hold time (max dur between osc msgs), 1 -> release time]
	env.scope;
	env = SinOsc.ar(60*(In.kr(bus).lag(0.5)*50).midiratio) * env;
	Splay.ar(env);
}.play;
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




