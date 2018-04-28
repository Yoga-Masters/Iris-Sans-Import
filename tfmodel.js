// Setup data into the right variables; change this to parameters in train model functions
var CLASSES = ANGLES_MAGNITUDES_CLASSES;
var NUM_CLASSES = ANGLES_MAGNITUDES_NUM_CLASSES;
var DATA = ANGLES_MAGNITUDES_DATA;
var DATA_LENGTH = DATA[0].length - 1;
// Logging of data
console.log("Labels:", CLASSES, "=>", NUM_CLASSES, "Labels");
console.log("Angles_Magnitudes; Shape:", [1, DATA_LENGTH]);
console.log("Data:", DATA);
const trainingConsts = [0.01, 0.001, 2000]; // Test split, 0 for maximum training; learning rate & epochs
//Calling the train model function and logging the result
let model;
trainModel();
console.log(model);
//Two functions to use for the app below; one to train, one to predict
function trainModel() {
    const [xTrain, yTrain, xTest, yTest] = getPreppedTrainingData(0.15, DATA, CLASSES, NUM_CLASSES);

}

function getConfidences(data) {

}
//All functions below used as tensorflow library
function getPreppedTrainingData(testSplit, data, classes, numClasses) {
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