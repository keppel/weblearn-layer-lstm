## weblearn-layer-lstm

Long short-term memory layer for [WebLearn].

### Usage
```
npm install weblearn-layer-lstm
```

```js
const LSTM = require('weblearn-layer-lstm')

const lstm = LSTM(inputVectorDimension, numHiddenUnits)

const h = lstm.forward(x)
const grad_x = lstm.backward(x, grad_h)
```

Ported to WebLearn from [@jcjohnson](https://github.com/jcjohnson)'s [torch-rnn](https://github.com/jcjohnson/torch-rnn)

[WebLearn]: https://github.com/keppel/weblearn
