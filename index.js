/***
*
* https://blog.webkid.io/neural-networks-in-javascript/
*
*/

const mnist = require('mnist'); 

const set = mnist.set(700, 20);

const trainingSet = set.training;
const testSet = set.test;

console.log("test done");
/***/

const synaptic = require('synaptic');

const Layer = synaptic.Layer;
const Network = synaptic.Network;
const Trainer = synaptic.Trainer;

const inputLayer = new Layer(784);
const hiddenLayer = new Layer(100);
const outputLayer = new Layer(10);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer
});

console.log("network done");
/***/
const trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
    rate: .2,
    iterations: 20,
    error: .1,
    shuffle: true,
    log: 1,
    cost: Trainer.cost.CROSS_ENTROPY
});
console.log("trainer done");

console.log(myNetwork.activate(testSet[0].input));
console.log(testSet[0].output);