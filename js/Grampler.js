define(["Tone/core/Tone", "Tone/source/GrainPlayer", "Tone/component/AmplitudeEnvelope", "Tone/instrument/Instrument"],
function(Tone){

	"use strict";

	/**
	 *  @class Sampler wraps Tone.GrainPlayer in an AmplitudeEnvelope.
	 *
	 *  @constructor
	 *  @extends {Tone.Instrument}
	 *  @param {String} url the url of the audio file
	 *  @param {Function=} onload The callback to invoke when the sample is loaded.
	 *  @example
	 * var sampler = new Sampler("./audio/casio/A1.mp3", function(){
	 * 	//repitch the sample down a half step
	 * 	sampler.triggerAttack(-1);
	 * }).toMaster();
	 */
	Tone.Grampler = function(){

		var options = this.optionsObject(arguments, ["url", "onload"], Tone.Grampler.defaults);
		Tone.Instrument.call(this, options);

		/**
		 *  The sample player.
		 *  @type {Tone.GrainPlayer}
		 */
		this.GrainPlayer = new Tone.GrainPlayer(options.url, options.onload);
		this.GrainPlayer.retrigger = true;

		/**
		 *  The amplitude envelope.
		 *  @type {Tone.AmplitudeEnvelope}
		 */
		this.envelope = new Tone.AmplitudeEnvelope(options.envelope);

		this.GrainPlayer.chain(this.envelope, this.output);
		this._readOnly(["grainplayer", "envelope"]);
		this.loop = options.loop;
		this.reverse = options.reverse;
	};

	Tone.extend(Tone.Grampler, Tone.Instrument);

	/**
	 *  the default parameters
	 *  @static
	 */
	Tone.Grampler.defaults = {
		"onload" : Tone.noOp,
		"loop" : false,
		"reverse" : false,
		"envelope" : {
			"attack" : 0.001,
			"decay" : 0,
			"sustain" : 1,
			"release" : 0.1
		}
	};

	/**
	 *  Trigger the start of the sample.
	 *  @param {Interval} [pitch=0] The amount the sample should
	 *                              be repitched.
	 *  @param {Time} [time=now] The time when the sample should start
	 *  @param {NormalRange} [velocity=1] The velocity of the note
	 *  @returns {Tone.Grampler} this
	 *  @example
	 * sampler.triggerAttack(0, "+0.1", 0.5);
	 */
	Tone.Grampler.prototype.triggerAttack = function(pitch, time, velocity){
		time = this.toSeconds(time);
		pitch = this.defaultArg(pitch, 0);
		this.GrainPlayer.playbackRate = this.intervalToFrequencyRatio(pitch);
		this.GrainPlayer.start(time);
		this.envelope.triggerAttack(time, velocity);
		return this;
	};

	/**
	 *  Start the release portion of the sample. Will stop the sample once the
	 *  envelope has fully released.
	 *
	 *  @param {Time} [time=now] The time when the note should release
	 *  @returns {Tone.Grampler} this
	 *  @example
	 * sampler.triggerRelease();
	 */
	Tone.Grampler.prototype.triggerRelease = function(time){
		time = this.toSeconds(time);
		this.envelope.triggerRelease(time);
		this.GrainPlayer.stop(this.toSeconds(this.envelope.release) + time);
		return this;
	};

	/**
	 * If the output sample should loop or not.
	 * @memberOf Tone.Grampler#
	 * @type {number|string}
	 * @name loop
	 */
	Object.defineProperty(Tone.Grampler.prototype, "loop", {
		get : function(){
			return this.GrainPlayer.loop;
		},
		set : function(loop){
			this.GrainPlayer.loop = loop;
		}
	});

	/**
	 * The direction the buffer should play in
	 * @memberOf Tone.Grampler#
	 * @type {boolean}
	 * @name reverse
	 */
	Object.defineProperty(Tone.Grampler.prototype, "reverse", {
		get : function(){
			return this.GrainPlayer.reverse;
		},
		set : function(rev){
			this.GrainPlayer.reverse = rev;
		}
	});

	/**
	 *  Clean up.
	 *  @returns {Tone.Grampler} this
	 */
	Tone.Grampler.prototype.dispose = function(){
		Tone.Instrument.prototype.dispose.call(this);
		this._writable(["player", "envelope"]);
		this.GrainPlayer.dispose();
		this.GrainPlayer = null;
		this.envelope.dispose();
		this.envelope = null;
		return this;
	};

	return Tone.Grampler;
});
