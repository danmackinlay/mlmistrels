(
n=8;
// alloc a buffer to playback
s.waitForBoot ({
	b = Buffer.alloc(s, s.sampleRate * 60.0, 1);
	SynthDef(\grain,
		{arg out=0,
			rate=1.0,
			amp=0.25,
			pos=0,
			attackTime=0.05,
			releaseTime=1.0,
			interp=2,
			overlaps=4,
			windowRandRatio=0.25,
			pan=0.5,
			bufnum=0,
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
				bufnum: bufnum,
				pointer: pos,
				freqScale: rate,
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
OSCdef.newMatching(\file, {|msg, time, addr, recvPort|
	[\file, msg].postln;
	//endlessly allocate new buffers because otherwise doesn't load right
	b= MP3.readToBuffer(s, msg[1]);
}, '/file');
OSCdef.newMatching(\note, {|msg, time, addr, recvPort|
	var pos, gain, rate;
	var path = msg.removeAt(0);
	pos = msg.copyRange(0, n-1);
	gain = msg.copyRange(n, 2*n-1);
	rate = msg.copyRange(2*n, 3*n);
	[\note, n, pos, gain, rate].postln;
	n.do({|i|
		var thisRate = rate[i], thisPos = pos[i], thisAmp=gain[i];
		[\subnote, i, thisRate, thisPos, thisAmp].postln;
		Synth(\grain, [
			\rate, thisRate,
			\amp, thisAmp,
			\pos, thisPos,
			\bufnum, b
		]);
	});
}, '/note');
)
OSCdef.trace(true, true);

b.plot;
b= MP3.readToBuffer(s, '/Users/dan/Source/mlmistrels/samples/clarinet.mp3');