// ===================== ^^^ DATA JS VARIABLES LOAD IN ^^^ ======================
// ============================= GLOBAL VARIABLES ===============================
var selectedData = document.getElementById("datatype");
const ALL_DATA = {
    "CLASSES": ["warriorii", "tree", "triangle"],
    "NUM_CLASSES": 3,
    "datatype0": DEFAULT_DATA,
    "datatype1": MAGNITUDES_DATA,
    "datatype2": POSITIONS_DATA,
    "datatype3": ANGLES_DATA,
    "datatype4": ANGLES_MAGNITUDES_DATA
}
IRIS_CLASSES = ALL_DATA["CLASSES"];
IRIS_NUM_CLASSES = ALL_DATA["NUM_CLASSES"];
IRIS_DATA = ALL_DATA[selectedData.value];
var IRIS_DATA_LENGTH = IRIS_DATA[0].length - 1;
let model;
// ============================== MAIN FUNCTIONS ================================
// The main function of the Iris demo.
async function iris() {
    document.getElementById('train-from-scratch').addEventListener('click', async () => {
        IRIS_DATA = ALL_DATA[selectedData.value];
        IRIS_DATA_LENGTH = IRIS_DATA[0].length - 1;
        const [xTrain, yTrain, xTest, yTest] = getIrisData(0.01);
        model = await trainModel(xTrain, yTrain, xTest, yTest);
        evaluateModelOnTestData(model, xTest, yTest);
    });
    status('Standing by for datatype ' + selectedData.value);
    wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}
iris();
// ============================ DATAUSE JS FUNCTIONS ============================
/**
 * Obtains Iris data, split into training and test sets.
 *
 * @param testSplit Fraction of the data at the end to split as test data: a
 *   number between 0 and 1.
 *
 * @param return A length-IRIS_DATA_LENGTH `Array`, with
 *   - training data as an `Array` of length-IRIS_DATA_LENGTH `Array` of numbers.
 *   - training labels as an `Array` of numbers, with the same length as the
 *     return training data above. Each element of the `Array` is from the set
 *     {0, 1, 2}.
 *   - test data as an `Array` of length-IRIS_DATA_LENGTH `Array` of numbers.
 *   - test labels as an `Array` of numbers, with the same length as the
 *     return test data above. Each element of the `Array` is from the set
 *     {0, 1, 2}.
 */
function getIrisData(testSplit) {
    return tf.tidy(() => {
        const dataByClass = [];
        const targetsByClass = [];
        for (let i = 0; i < IRIS_CLASSES.length; ++i) {
            dataByClass.push([]);
            targetsByClass.push([]);
        }
        for (const example of IRIS_DATA) {
            const target = example[example.length - 1];
            const data = example.slice(0, example.length - 1);
            dataByClass[target].push(data);
            targetsByClass[target].push(target);
        }
        const xTrains = [];
        const yTrains = [];
        const xTests = [];
        const yTests = [];
        for (let i = 0; i < IRIS_CLASSES.length; ++i) {
            const [xTrain, yTrain, xTest, yTest] =
            convertToTensors(dataByClass[i], targetsByClass[i], testSplit);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xTests.push(xTest);
            yTests.push(yTest);
        }

        const concatAxis = 0;
        return [
            tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
            tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        ];
    });
}

/**
 * Convert Iris data arrays to `tf.Tensor`s.
 *
 * @param data The Iris input feature data, an `Array` of `Array`s, each element
 *   of which is assumed to be a length-IRIS_DATA_LENGTH `Array` (for petal length, petal
 *   width, sepal length, sepal width).
 * @param targets An `Array` of numbers, with values from the set {0, 1, 2}:
 *   representing the true category of the Iris flower. Assumed to have the same
 *   array length as `data`.
 * @param testSplit Fraction of the data at the end to split as test data: a
 *   number between 0 and 1.
 * @return A length-IRIS_DATA_LENGTH `Array`, with
 *   - training data as `tf.Tensor` of shape [numTrainExapmles, IRIS_DATA_LENGTH].
 *   - training one-hot labels as a `tf.Tensor` of shape [numTrainExamples, 3]
 *   - test data as `tf.Tensor` of shape [numTestExamples, IRIS_DATA_LENGTH].
 *   - test one-hot labels as a `tf.Tensor` of shape [numTestExamples, 3]
 */
function convertToTensors(data, targets, testSplit) {
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and split have different numbers of examples');
    }

    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;

    const xDims = data[0].length;

    // Create a 2D `tf.Tensor` to hold the feature data.
    const xs = tf.tensor2d(data, [numExamples, xDims]);

    // Create a 1D `tf.Tensor` to hold the labels, and convert the number label
    // from the set {0, 1, 2} into one-hot encoding (.e.g., 0 --> [1, 0, 0]).
    const ys = tf.oneHot(tf.tensor1d(targets), IRIS_NUM_CLASSES);

    // Split the data into training and test sets, using `slice`.
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, IRIS_NUM_CLASSES]);
    const yTest = ys.slice([0, 0], [numTestExamples, IRIS_NUM_CLASSES]);
    return [xTrain, yTrain, xTest, yTest];
}
// ============================= INDEX JS FUNCTIONS =============================
/**
 * Train a `tf.Model` to recognize Iris flower type.
 *
 * @param xTrain Training feature data, a `tf.Tensor` of shape
 *   [numTrainExamples, IRIS_DATA_LENGTH]. The second dimension include the features
 *   petal length, petalwidth, sepal length and sepal width.
 * @param yTrain One-hot training labels, a `tf.Tensor` of shape
 *   [numTrainExamples, 3].
 * @param xTest Test feature data, a `tf.Tensor` of shape [numTestExamples, IRIS_DATA_LENGTH].
 * @param yTest One-hot test labels, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 * @returns The trained `tf.Model` instance.
 */
async function trainModel(xTrain, yTrain, xTest, yTest) {
    var time = Date.now();
    status("Training model @ " + (new Date(time)).toLocaleTimeString() + "... Please wait.");
    const params = loadTrainParametersFromUI();
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
    // Call `model.fit` to train the model.
    const history = await model.fit(xTrain, yTrain, {
        epochs: params.epochs,
        validationData: [xTest, yTest],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Plot the loss and accuracy values at the end of every training epoch.
                if (showGraphs.checked) {
                    plotLosses(lossValues, epoch, logs.loss, logs.val_loss);
                    plotAccuracies(accuracyValues, epoch, logs.acc, logs.val_acc);
                }
                // Await web page DOM to refresh for the most recently plotted values.
                await tf.nextFrame();
            },
        }
    });
    status("Model training complete @ " + (new Date(Date.now())).toLocaleTimeString() + ", in " + (Date.now() - time) + " ms; AKA: " + convertMS(Date.now() - time).m + " mins " + convertMS(Date.now() - time).s + " seconds.");
    return model;
}

/**
 * Run inference on manually-input Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 */
async function predictOnManualInput(model) {
    if (model == null) setManualInputWinnerMessage('ERROR: Please load or train model first.');
    else tf.tidy(() => { // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the `predict` call is released at the end.
        // Prepare input data as a 2D `tf.Tensor`.
        const inputData = getManualInputData();
        const input = tf.tensor2d([inputData], [1, IRIS_DATA_LENGTH]);
        // Call `model.predict` to get the prediction output as probabilities for
        // the Iris flower categories.
        const predictOut = model.predict(input);
        const logits = Array.from(predictOut.dataSync());
        const winner = IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
        setManualInputWinnerMessage(winner);
        renderLogitsForManualInput(logits);
    });
}

/**
 * Run inference on some test Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 * @param xTest Test data feature, a `tf.Tensor` of shape [numTestExamples, IRIS_DATA_LENGTH].
 * @param yTest Test true labels, one-hot encoded, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 */
async function evaluateModelOnTestData(model, xTest, yTest) {
    clearEvaluateTable();
    tf.tidy(() => {
        const xData = xTest.dataSync();
        const yTrue = yTest.argMax(-1).dataSync();
        const predictOut = model.predict(xTest);
        const yPred = predictOut.argMax(-1);
        renderEvaluateTable(
            xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    });
    predictOnManualInput(model);
}
// =============================== UI JS FUNCTIONS ==============================
/**
 * Clear the evaluation table.
 */
function clearEvaluateTable() {
    const tableBody = document.getElementById('evaluate-tbody');
    while (tableBody.children.length > 1) {
        tableBody.removeChild(tableBody.children[1]);
    }
}

/**
 * Plot new loss values.
 *
 * @param lossValues An `Array` of data to append to.
 * @param epoch Training epoch number.
 * @param newTrainLoss The new training loss, as a single `Number`.
 * @param newValidationLoss The new validation loss, as a single `Number`.
 */
function plotLosses(lossValues, epoch, newTrainLoss, newValidationLoss) {
    lossValues.push({
        'epoch': epoch,
        'loss': newTrainLoss,
        'set': 'train'
    });
    lossValues.push({
        'epoch': epoch,
        'loss': newValidationLoss,
        'set': 'validation'
    });
    vegaEmbed(
        '#lossCanvas', {
            '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
            'data': {
                'values': lossValues
            },
            'mark': 'line',
            'encoding': {
                'x': {
                    'field': 'epoch',
                    'type': 'ordinal'
                },
                'y': {
                    'field': 'loss',
                    'type': 'quantitative'
                },
                'color': {
                    'field': 'set',
                    'type': 'nominal'
                },
            }
        }, {});
}

/**
 * Plot new accuracy values.
 *
 * @param lossValues An `Array` of data to append to.
 * @param epoch Training epoch number.
 * @param newTrainLoss The new training accuracy, as a single `Number`.
 * @param newValidationLoss The new validation accuracy, as a single `Number`.
 */
function plotAccuracies(
    accuracyValues, epoch, newTrainAccuracy, newValidationAccuracy) {
    accuracyValues.push({
        'epoch': epoch,
        'accuracy': newTrainAccuracy,
        'set': 'train'
    });
    accuracyValues.push({
        'epoch': epoch,
        'accuracy': newValidationAccuracy,
        'set': 'validation'
    });
    vegaEmbed(
        '#accuracyCanvas', {
            '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
            'data': {
                'values': accuracyValues
            },
            'mark': 'line',
            'encoding': {
                'x': {
                    'field': 'epoch',
                    'type': 'ordinal'
                },
                'y': {
                    'field': 'accuracy',
                    'type': 'quantitative'
                },
                'color': {
                    'field': 'set',
                    'type': 'nominal'
                },
            }
        }, {});
}

/**
 * Get manually input Iris data from the input boxes.
 */
function getManualInputData() {
    var manualInput = [];
    for (i = 1; i <= IRIS_DATA_LENGTH; i++) manualInput.push(Number(document.getElementById('data' + i).value));
    return manualInput;
}

function setManualInputWinnerMessage(message) {
    const winnerElement = document.getElementById('winner');
    winnerElement.textContent = message;
}

function logitsToSpans(logits) {
    let idxMax = -1;
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < logits.length; ++i) {
        if (logits[i] > maxLogit) {
            maxLogit = logits[i];
            idxMax = i;
        }
    }
    const spans = [];
    for (let i = 0; i < logits.length; ++i) {
        const logitSpan = document.createElement('span');
        logitSpan.textContent = logits[i].toFixed(3);
        if (i === idxMax) {
            logitSpan.style['font-weight'] = 'bold';
        }
        logitSpan.classList = ['logit-span'];
        spans.push(logitSpan);
    }
    return spans;
}

function renderLogits(logits, parentElement) {
    while (parentElement.firstChild) {
        parentElement.removeChild(parentElement.firstChild);
    }
    logitsToSpans(logits).map(logitSpan => {
        parentElement.appendChild(logitSpan);
    });
}

function renderLogitsForManualInput(logits) {
    const logitsElement = document.getElementById('logits');
    renderLogits(logits, logitsElement);
}

function renderEvaluateTable(xData, yTrue, yPred, logits) {
    const tableBody = document.getElementById('evaluate-tbody');
    for (let i = 0; i < yTrue.length; ++i) {
        const row = document.createElement('tr');
        for (let j = 0; j < IRIS_DATA_LENGTH; ++j) {
            const cell = document.createElement('td');
            cell.textContent = xData[IRIS_DATA_LENGTH * i + j].toFixed(1);
            row.appendChild(cell);
        }
        const truthCell = document.createElement('td');
        truthCell.textContent = IRIS_CLASSES[yTrue[i]];
        row.appendChild(truthCell);
        const predCell = document.createElement('td');
        predCell.textContent = IRIS_CLASSES[yPred[i]];
        predCell.classList =
            yPred[i] === yTrue[i] ? ['correct-prediction'] : ['wrong-prediction'];
        row.appendChild(predCell);
        const logitsCell = document.createElement('td');
        const exampleLogits =
            logits.slice(i * IRIS_NUM_CLASSES, (i + 1) * IRIS_NUM_CLASSES);
        logitsToSpans(exampleLogits).map(logitSpan => {
            logitsCell.appendChild(logitSpan);
        });
        row.appendChild(logitsCell);
        tableBody.appendChild(row);
    }
}

function wireUpEvaluateTableCallbacks(predictOnManualInputCallback) {
    for (i = 1; i <= IRIS_DATA_LENGTH; i++) document.getElementById('data' + i).addEventListener('change', () => {
        status('Standing by for datatype ' + selectedData.value);
        wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
        predictOnManualInputCallback();
    });
}

function loadTrainParametersFromUI() {
    return {
        epochs: Number(document.getElementById('train-epochs').value),
        learningRate: Number(document.getElementById('learning-rate').value)
    };
}

function status(statusText) {
    console.log(statusText);
    document.getElementById('demo-status').textContent = statusText;
}

function disableLoadModelButtons() {
    document.getElementById('load-pretrained-remote').style.display = 'none';
    document.getElementById('load-pretrained-local').style.display = 'none';
}
// ============================= LOADER JS FUNCTIONS ============================
/**
 * Test whether a given URL is retrievable.
 */
async function urlExists(url) {
    status('Testing url ' + url);
    try {
        const response = await fetch(url, {
            method: 'HEAD'
        });
        return response.ok;
    } catch (err) {
        return false;
    }
}

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
async function loadHostedPretrainedModel(url) {
    status('Loading pretrained model from ' + url);
    try {
        const model = await tf.loadModel(url);
        status('Done loading pretrained model.');
        // We can't load a model twice due to
        // https://github.com/tensorflow/tfjs/issues/34
        // Therefore we remove the load buttons to avoid user confusion.
        disableLoadModelButtons();
        return model;
    } catch (err) {
        console.error(err);
        status('Loading pretrained model failed.');
    }
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