// Importing required modules
const cv = require('opencv4nodejs');
const mp = require('@mediapipe/face_mesh');
const math = require('mathjs');

// Initializing variables
const mp_face_mesh = mp.solutions.face_mesh;
const face_cascade = new cv.CascadeClassifier('haarcascade_frontalface_alt.xml');

const LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_IRIS = [469, 470, 471, 472];
const L_H_LEFT = [33];
const L_H_RIGHT = [133];
const R_H_LEFT = [362];
const R_H_RIGHT = [263];

// Function to calculate Euclidean distance
function euclidian_distance(vec1, vec2) {
  const [x1, y1] = vec1.ravel();
  const [x2, y2] = vec2.ravel();
  const distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  return distance;
}

// Function to determine eye position
function position(iris_center, right_vec, left_vec) {
  const center_right_distance = euclidian_distance(iris_center, right_vec);
  const left_right_distance = euclidian_distance(right_vec, left_vec);
  const ratio = center_right_distance / left_right_distance;
  if (ratio <= 1.18) {
    position = 'left';
  } else if (ratio > 1.20 && ratio <= 1.33) {
    position = 'center';
  } else {
    position = 'right';
  }
  return [position, ratio];
}

// Accessing the camera to capture video
const cap = new cv.VideoCapture(1);

// Initializing the face mesh
const face_mesh = new mp_face_mesh.FaceMesh({
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Continuously reading frames from the camera
while (true) {
  let frame = cap.read();
  if (frame.empty) {
    break;
  }
  frame = cv.flip(frame, 1);
  const rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
  const faces = face_cascade.detectMultiScale(rgb_frame, 1.1, 5, new cv.Size(25, 25));
  if (faces.length > 1) {
    console.log("The student must take the test alone!!!!!!!");
  }

  const [img_h, img_w] = frame.sizes;
  const results = face_mesh.process(rgb_frame);

  if (results.multiFaceLandmarks) {
    const mesh_points = results.multiFaceLandmarks[0].landmarks.map(p => {
      return [p.x * img_w, p.y * img_h];
    });
    const l_iris = cv.minEnclosingCircle(new cv.Mat(mesh_points.slice(474, 478)));
    const r_iris = cv.minEnclosingCircle(new cv.Mat(mesh_points.slice(469, 473)));
    const center_left = new cv.Point
