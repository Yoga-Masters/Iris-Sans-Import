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