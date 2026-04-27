class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / this.targetRate;
    this.acc = [];
    this.accSize = 0;
    this.flushEvery = Math.round(this.targetRate * 0.1);
    this.leftover = new Float32Array(0);
    this.pos = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];

    const buf = new Float32Array(this.leftover.length + samples.length);
    buf.set(this.leftover);
    buf.set(samples, this.leftover.length);

    let idx = 0;
    const out = [];
    while (this.pos < buf.length - 1) {
      const i0 = Math.floor(this.pos);
      const frac = this.pos - i0;
      out.push(buf[i0] * (1 - frac) + buf[i0 + 1] * frac);
      this.pos += this.ratio;
    }

    if (out.length > 0) {
      const outArr = new Float32Array(out);
      this.acc.push(outArr);
      this.accSize += outArr.length;
    }

    const consume = Math.floor(this.pos);
    this.leftover = buf.slice(consume);
    this.pos -= consume;

    if (this.accSize >= this.flushEvery) {
      const merged = new Float32Array(this.accSize);
      let offset = 0;
      for (const c of this.acc) { merged.set(c, offset); offset += c.length; }
      this.acc = [];
      this.accSize = 0;

      const pcm = new Int16Array(merged.length);
      let sumSq = 0;
      for (let i = 0; i < merged.length; i++) {
        const s = Math.max(-1, Math.min(1, merged[i]));
        pcm[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7FFF);
        sumSq += s * s;
      }
      const rms = Math.sqrt(sumSq / merged.length);

      this.port.postMessage({ pcm: pcm.buffer, rms }, [pcm.buffer]);
    }

    return true;
  }
}
registerProcessor('pcm-downsampler', PCMDownsampler);
