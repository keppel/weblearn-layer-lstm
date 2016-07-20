const Tensor = require('weblearn-tensor')
const Module = require('weblearn-module')
const assert = require('assert')
const old = require('old')

/*
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceLSTM stores this many
scalar values:
NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H
For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
 */

class LSTM extends Module {
  constructor(inputDim, hiddenDim) {
    super()

    const D = inputDim
    const H = hiddenDim

    this.inputDim = D
    this.hiddenDim = hiddenDim

    this.weight = Tensor(D + H, 4 * H)
    this.gradWeight = Tensor(D + H, 4 * H).zero()
    this.bias = Tensor(4 * H)
    this.gradBias = Tensor(4 * H).zero()
    this.reset()

    this.cell = Tensor()   // (N, T, H)
    this.gates = Tensor()  // (N, T, 4H)
    this.buffer1 = Tensor()// (N, H)
    this.buffer2 = Tensor()// (N, H)
    this.buffer3 = Tensor()// (1, 4H)
    this.gradABuffer = Tensor() // (N, 4H)

    this.h0 = Tensor()
    this.c0 = Tensor()
    this.rememberStates = false

    this.gradc0 = Tensor()
    this.gradh0 = Tensor()
    this.gradx = Tensor()
    this.gradInput = [this.gradc0, this.gradh0, this.gradx]
  }

  reset(std) {
    if(!std) {
      std = 1 / Math.sqrt(this.hiddenDim + this.inputDim)
    }
    this.bias.zero()
    this.bias.slice([[this.hiddenDim + 1, 2 * this.hiddenDim]])
    this.weight.normal(0, std)
    return this
  }

  resetStates() {
    this.h0 = this.h0.new()
    this.c0 = this.c0.new()
  }

  

  _unpackInput(input) {
    let c0, h0, x
    if(!input.values && input.length === 3){
      [c0, h0, x] = input 
    }
    else if(!input.values && input.length === 2){
      [h0, x] = input
    } else if(input.values) {
      // input is a tensor
      x = input
    } else {
      assert(false, 'invalid input')
    }
    return [c0, h0, x]
  }

  _getSizes(input, gradOutput) { 
    const [c0, h0, x] = this._unpackInput(input)
    const N = x.size(0)
    const T = x.size(1)
    const H = this.hiddenDim
    const D = this.inputDim

    checkDims(x, [N, T, D])
    if(h0){
      checkDims(h0, [N, H])
    }
    if(c0){
      checkDims(c0, [N, H])
    }
    if(gradOutput){
      checkDims(gradOutput, [N, T, H])
    }
    return [N, T, D, H]
  }

  /*
  Input:
  - c0: Initial cell state, (N, H)
  - h0: Initial hidden state, (N, H)
  - x: Input sequence, (N, T, D)
  Output:
  - h: Sequence of hidden states, (N, T, H)
   */

  updateOutput(input) {
    this.recomputeBackward = true
    let [c0, h0, x] = this._unpackInput(input)
    const [N, T, D, H] = this._getSizes(input)

    this._returnGradc0 = !!c0
    this._returnGradh0 = !!h0
    if(!c0){
      c0 = this.c0
      if(c0.nElement() === 0 || !this.rememberStates){
        c0.resize(N, H).zero()
      } else if(this.rememberStates) {
        const prevN = this.cell.size(0)
        const prevT = this.cell.size(1)
        assert(prevN === N, 'batch sizes must be constant to remember states')
        c0.copy(this.cell.slice([[], prevT]))
      }
    }

    if(!h0){
      h0 = this.h0
      if(h0.nElement() === 0 || !this.rememberStates){
        h0.resize(N, H).zero()
      } else if(this.rememberStates){
        const prevN = this.cell.size(0)
        const prevT = this.cell.size(1)
        assert(prevN === N, 'batch sizes must be constant to remember states')
        h0.copy(this.output.slice([[], prevT]))
      }
    }

    const biasExpand = this.bias.view(1, 4 * H).expand(N, 4 * H) 
    const Wx = this.weight.slice([[1, D]])
    const Wh = this.weight.slice([[D + 1, D + H]])

    let h = this.output
    let c = this.cell
    h.resize(N, T, H).zero()
    c.resize(N, T, H).zero()

    let prevh = h0
    let prevc = c0
    this.gates.resize(N, T, 4 * H).zero()

    for(let t = 0; t < T - 1; t++){ // off-by-one?
      const curx = x.slice([[], t])
      const nexth = h.slice([[], t])
      const nextc = c.slice([[], t])

      let curGates = this.gates.slice([[], t])
      curGates.addmm(biasExpand, curx, Wx)
      curGates.addmm(prevh, Wh)
      curGates.slice([[], [1, 3 * H]]).sigmoid()
      curGates.slice([[], [3 * H + 1, 4 * H]]).tanh()
      const i = curGates.slice([[], [1, H]])
      const f = curGates.slice([[], [H + 1, 2 * H]])
      const o = curGates.slice([[], [2 * H + 1, 3 * H]])
      const g = curGates.slice([[], [3 * H + 1, 4 * H]])
      nexth.cmul(i, g)
      nextc.cmul(f, prev_c).add(nexth)
      nexth.tanh(nextc).cmul(o)
      prevh = nexth
      prevc = nextc
    }

    return this.output
  }

  backward(input, gradOutput, scale) {
    this.recomputeBackward = false
    scale = scale || 1
    assert(scale === 1, 'must have scale=1')
    let [c0, h0, x] = this._unpackInput(input)
    if(!c0){c0 = this.c0}
    if(!h0){h0 = this.h0}

    let gradc0 = this.gradc0
    let gradh0 = this.gradh0
    let gradx = this.gradx
    let h = this.output
    let c = this.cell
    let gradh = gradOutput

    const [N, T, D, H] = this._getSizes(input, gradOutput)
    const Wx = this.weight.slice([[1, D]])
    const Wh = this.weight.slice([[D + 1, D + H]])
    const gradWx = this.gradWeight.slice([[1, D]])
    const gradWh = this.gradWeight.slice([[D + 1, D + H]])
    const gradb = this.gradBias

    gradh0.resizeAs(h0).zero()
    gradc0.resizeAs(c0).zero()
    gradx.resizeAs(x).zero()
    let gradNexth = this.buffer1.resizeAs(h0).zero()
    let gradNextc = this.buffer2.resizeAs(c0).zero()
    for(let t = T; t >= 0; t--){
      let nexth = h.slice([[], t])
      let nextc = c.slice([[], t])
      let prevh, prevc

      if(t === 0) {
        prevh = h0
        prevc = c0
      } else {
        prevh = h.slice([[], t - 1])
        prevc = c.slice([[], t - 1])
      }
      gradNexth.add(gradh.slice([[], t]))

      let i = this.gates.slice([[], t, [0, H]])
      let f = this.gates.slice([[], t, [H + 1, 2 * H]])
      let o = this.gates.slice([[], t, [2 * H + 1, 3 * H]])
      let g = this.gates.slice([[], t, [3 * H + 1, 4 * H]])

      let grada = this.gradABuffer.resize(N, 4 * H).zero()
      let gradai = grada.slice([[], [1, H]])
      let gradaf = grada.slice([[], [H + 1, 2 * H]])
      let gradao = grada.slice([[], [2 * H + 1, 3 * H]])
      let gradag = grada.slice([[], [3 * H + 1, 4 * H]])

      /*
      We will use gradai, gradaf, and gradao as temporary buffers
      to to compute gradnextc. We will need tanhnextc (stored in gradai)
      to compute gradao; the other values can be overwritten after we compute
      gradnextc
      */
      let tanhNextc = gradai.tanh(nextc)
      let tanhNextc2 = gradaf.cmul(tanhNextc, tanhNextc)
      let myGradNextc = gradao
      myGradNextc.fill(1).add(-1, tanhNextc2).cmul(o).cmul(gradNexth)
      gradNextC.add(myGradNextc)

      // We need tanhNextC(currently in gradai) to compute gradao; after
      // that we can overwrite it.
      gradao.fill(1).add(-1, o).cmul(o).cmul(tanhNextc).cmul(gradNexth)

      // Use gradai as a temporary buffer for computing gradag
      let g2 = gradai.cmul(g, g)
      gradag.fill(1).add(-1, g2).cmul(i).cmul(gradNextc)

      // We don't need any temporary storage for these so do them last
      gradai.fill(1).add(-1, i).cmul(i).cmul(g).cmul(gradNextc)
      gradaf.fill(1).add(-1, f).cmul(f).cmul(prevc).cmul(gradNextc)

      gradx.slice([[], t]).mm(grada, Wx.t())
      gradWx.addmm(scale, x.slice([[], t]).t(), grada)
      gradWh.addmm(scale, prevh.t(), grada)
      let gradaSum = this.buffer3.resize(1, 4 * H).sum(grada, 1)
      gradb.add(scale, gradaSum)

      gradNexth.mm(grada, Wh.t())
      gradNextc.cmul(f)
    }

    gradh0.copy(gradNexth)
    gradc0.copy(gradNextc)

    if(this._returnGradc0 && this._returnGradh0){
      this.gradInput = [this.gradc0, this.gradh0, this.gradx]
    } else if(this._returnGradh0){
      this.gradInput = [this.gradh0, this.gradx]
    } else {
      this.gradInput = this.gradx
    }

    return this.gradInput
  }

  clearState() {
    this.cell.set()
    this.gates.set()
    this.buffer1.set()
    this.buffer2.set()
    this.buffer3.set()
    this.gradABuffer.set()

    this.gradc0.set()
    this.gradh0.set()
    this.gradx.set()
    this.output.set()
  }

  updateGradInput(input, gradOutput) {
    if(this.recomputeBackward){
      this.backward(input, gradOutput, 1)
    }
    return this.gradInput
  }

  accGradParameters(input, gradOutput, scale) {
    if(this.recomputeBackward){
      this.backward(input, gradOutput, scale)
    }
  }

}

function checkDims(x, dims) {
  assert(x.dim() === dims.length)
  Object.keys(dims).forEach(k => {
    assert(x.size(k) === dims[k])
  })
}


module.exports = old(LSTM)