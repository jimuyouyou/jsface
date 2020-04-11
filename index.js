const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');
// const wfetch = require('whatwg-fetch');
// const { fetch } = wfetch;
const fetch = require('node-fetch');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
faceapi.env.monkeyPatch({ fetch: fetch });

const testImage = async () => {
  // console.log(myimg);

  try {
    // console.log(faceapi.nets);
    const myimg = await canvas.loadImage('./girl.jpeg');
    console.log('loading models from network, please wait few mintutes');
    await faceapi.nets.tinyFaceDetector.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json');
    await faceapi.nets.ageGenderNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/age_gender_model-weights_manifest.json');
    console.time('aa');
    const results = await faceapi.detectSingleFace(myimg, new faceapi.TinyFaceDetectorOptions({ inputSize: 1280 })).withAgeAndGender();
    console.timeEnd('aa');
    console.log(results);
  } catch (e) {
    console.log('e', e);
  }
};

testImage();