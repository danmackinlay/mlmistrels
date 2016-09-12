(
// alloc a buffer to playback
s.waitForBoot ({
	b = Buffer.alloc(s, s.sampleRate * 60.0, 1);
	n=8;
	SynthDef(\snip,
		{arg out=0,
			pitchRatio=1.0,
			amp=0.25,
			pos=0,
			attackTime=0.05,
			releaseTime=1.0,
			interp=2,
			overlaps=4,
			windowRandRatio=0.25,
			pan=0.5,
			grainsize=0.1;
			var env, sig;
			env = EnvGen.kr(
				Env.perc(attackTime, releaseTime),
				levelScale: amp,
				gate: 1,
				doneAction: 2
			);
			sig = Warp1.ar(
				numChannels: 1,
				bufnum: b,
				pointer: pos,
				freqScale: pitchRatio,
				interp: interp,
				windowSize: grainsize,
				overlaps: overlaps,
				windowRandRatio: windowRandRatio,
				mul: env
			);
			Out.ar(out, sig);
		},
		[],
	).add;

});
n = NetAddr("127.0.0.1", 57120); // local machine
OSCdef.newMatching(\file, {|msg, time, addr, recvPort| [\file, msg].postln}, '/file', n);
OSCdef.newMatching(\note, {|msg, time, addr, recvPort| [\note, msg].postln}, '/note', n);
)
