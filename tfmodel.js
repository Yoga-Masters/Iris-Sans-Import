// Setup data into the right variables; change this to parameters in train model functions
const trainingData = [ANGLES_MAGNITUDES_CLASSES, ANGLES_MAGNITUDES_DATA]; // Training data in one big data array
// trainingConsts: Test split, 0 for maximum training; learning rate & epochs
const slowTrainingConfig = [0.01, 0.001, 2000]; // Slow but almost guarenteed optimal learning with low loss
const fastTrainingConfig = [0.01, 0.03, 100]; // Rapid quickfire learning with high chance of high loss
//Calling the train model function and logging the result
let model;
getTrainedModel(trainingData, fastTrainingConfig).then(mdl => {
    model = mdl;
    console.log("MAIN RUNNER MODEL:", model);
    tryIt();
});

function tryIt() {
    var time = Date.now();
    getConfidences([90, -88, 98, -94, 103, -106, 17, -9, 65, -44, 6, -48, 0.2259, 0.2331, 0.3615, 0.3552, 0.5069, 0.5085, 0.1388, 0.1302, 0.383, 0.3689, 0.5681, 0.5769]).then(confs => {
        console.log("AFTER " + (Date.now() - time) + "MS, MAIN RUNNER OUTPUT:", confs);
    });
}

//Two functions to use for the app below; one to train, one to predict
function getTrainedModel(tData, config, cb) { // Returns a trained model
    return new Promise(function (resolve) {
        const classes = tData[0];
        const numClasses = tData[0].length;
        const data = tData[1];
        const [xTrain, yTrain, xTest, yTest] = getPreppedTrainingData(config[0], data, classes, numClasses);
        trainModel(xTrain, yTrain, xTest, yTest, {
            "epochs": config[2],
            "learningRate": config[1]
        }, (model) => {
            resolve(model);
        });
    });
}

function getConfidences(data) {
    return new Promise(function (resolve) {
        predictOnManualInput(model, data, trainingData[0], trainingData[1][0].length - 1, logits => {
            resolve(logits);
        });
    });
}
//All functions below used as tensorflow library
function getPreppedTrainingData(testSplit, data, classes, numClasses) {
    status("Prepping data for training with labels = " + JSON.stringify(classes) + ", length = " + numClasses + " with a " + testSplit * 100 + "% test-to-data split.");
    return tf.tidy(() => {
        const dataByClass = [];
        const targetsByClass = [];
        for (let i = 0; i < classes.length; ++i) {
            dataByClass.push([]);
            targetsByClass.push([]);
        }
        for (const example of data) {
            const target = example[example.length - 1];
            const data = example.slice(0, example.length - 1);
            dataByClass[target].push(data);
            targetsByClass[target].push(target);
        }
        const xTrains = [];
        const yTrains = [];
        const xTests = [];
        const yTests = [];
        for (let i = 0; i < classes.length; ++i) {
            const [xTrain, yTrain, xTest, yTest] = convertToTensors(dataByClass[i], targetsByClass[i], testSplit, numClasses);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xTests.push(xTest);
            yTests.push(yTest);
        }
        const concatAxis = 0;
        return [
            tf.concat(xTrains, concatAxis),
            tf.concat(yTrains, concatAxis),
            tf.concat(xTests, concatAxis),
            tf.concat(yTests, concatAxis)
        ];
    });
}

function trainModel(xTrain, yTrain, xTest, yTest, params, cb) {
    var time = Date.now();
    status("Training model @ " + (new Date(time)).toLocaleTimeString() + " on angles_magnitudes data. Training using a " + params.learningRate + " learning rate for " + params.epochs + " epochs; please wait...");
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        activation: 'sigmoid',
        inputShape: [xTrain.shape[1]]
    }));
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));
    const optimizer = tf.train.adam(params.learningRate);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    const lossValues = [];
    const accuracyValues = [];
    const history = model.fit(xTrain, yTrain, {
        epochs: params.epochs,
        validationData: [xTest, yTest]
    }).then(() => {
        status("Model training complete @ " + (new Date(Date.now())).toLocaleTimeString() + ", in " + (Date.now() - time) + " ms; AKA: " + convertMS(Date.now() - time).m + " mins " + convertMS(Date.now() - time).s + " seconds.");
        cb(model);
    });
}

function predictOnManualInput(model, inputData, classes, length, cb) {
    tf.tidy(() => {
        const input = tf.tensor2d([inputData], [1, length]);
        const predictOut = model.predict(input);
        const logits = Array.from(predictOut.dataSync()).map(x => Number.parseFloat(x).toFixed(3));
        logits.push(classes[predictOut.argMax(-1).dataSync()[0]]);
        cb(logits);
    });
}

function convertToTensors(data, targets, testSplit, numClasses) {
    const numExamples = data.length;
    if (numExamples !== targets.length) throw new Error('Data and Split have different # of examples');
    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;
    const xDims = data[0].length;
    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.oneHot(tf.tensor1d(targets), numClasses);
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, numClasses]);
    const yTest = ys.slice([0, 0], [numTestExamples, numClasses]);
    return [xTrain, yTrain, xTest, yTest];
}

function status(statusText) {
    console.log(statusText);
    document.getElementById('demo-status').textContent = statusText;
}

function convertMS(ms) {
    var d, h, m, s;
    s = Math.floor(ms / 1000);
    m = Math.floor(s / 60);
    s = s % 60;
    h = Math.floor(m / 60);
    m = m % 60;
    d = Math.floor(h / 24);
    h = h % 24;
    return {
        d: d,
        h: h,
        m: m,
        s: s
    };
}