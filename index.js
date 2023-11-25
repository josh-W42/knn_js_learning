/*
  K Nearest Neighbor Algorithm (KNN or k-NN)

  It is a non-parametric, supervised learning classifier.
  It uses proximity to make classifications or predictions about the
    grouping of an individual data point.

  The example used was:
    What if we wanted to predict what box a ball would fall into
    if it were dropped from a particular distance. (Like in the game, plinko)

    So we had a dependant variable, 1 box out of 10 that the ball could fall into.
    Independent variables,
      What pixel width is the ball being dropped? (0 - 500px)
      What is the ball's "bounciness" or elasticity.
      What is the size of the ball? (Pixel radius from center)

  So it was reasoned that, KNN could be used in this case to predict the output.

  To frame the problem, the data would be to constructed in a 2D array:
    array = [
      [drop_width, elasticity, ball_radius, box_ball_fell_into],
    ]
  
  And so, we picked a variable, drop_width. And framed our question like so:
    What box would a ball fall into if we dropped it at a drop_width of 300px?

  So we:
    - ran a simulation to retrieve some data.
    - Took the absolute value distance from each drop_point to the proposed position (300px).
    - Sorted the data by by the calculated distance in ascending order.
    - Took a slice of the data up until k rows.
    - Counted the number of occurrences of the outputs (box_ball_fell_into).
    - Segmented that data into pairs like ([box_4, occurred_2_times ])
    - Sorted again in ascending order.
    - Retrieved the value with the highest occurrences (the mode)
    - Retrieved the output.
    - Essentially that output would be your prediction result.

  I'll write this in code so it makes some more sense.
*/

// const _ = require("lodash");

// const outputs = [
//   [10, 0.5, 16, 1],
//   [200, 0.5, 16, 4],
//   [400, 0.5, 16, 4],
//   [400, 0.5, 16, 4],
// ];

// const prediction_drop_point = 300;
// const k = 3;

// const distance = (point) => {
//   return Math.abs(point - prediction_drop_point);
// };

// const runAnalysis = () => {
//   const prediction = _.chain(outputs)
//     .map((row) => [distance(row[0]), row[3]])
//     .sortBy((row) => row[0])
//     .slice(0, k)
//     .countBy((row) => row[1])
//     .toPairs()
//     .sortBy((row) => row[1])
//     .last()
//     .first()
//     .parseInt()
//     .value();
//   console.log(prediction);
// };

/**
 * Note that in practice this alone is not that accurate.
 * Here ares some changes we can apply to make this prediction more accurate:
 *  - Add more features. (Right now we're only observing the drop_width independent variable)
 *  - Change the prediction point. (varying values of drop_width)
 *  - Adjust parameters of analysis (Different values of k)
 *  - IF all else fails consider that there just isn't as strong of a correlation.
 *
 */

/**
 * Consider this: Before we even start making modifications to our algorithm,
 *    First lets develop a means to determine how accurate our algorithm is.
 *    One way of doing this is by: Finding an ideal k value
 *
 *  To find an ideal k value we had to run by a series of steps:
 *    - We record a bunch of data points.
 *    - We then split that data into a 'training' set and a 'test' set.
 *      - You may also want to shuffle the data as well.
 *    - In our test set we check each record by running KNN using the 'training' data.
 *    - After each test we ask: Does the KNN result equal the output in the 'test' record?
 *      If yes, then this is good. Our algorithm is working.
 *      If no, then we have a problem. We should remove this record from our test set.
 *    - After all of that we'll have to summarize the results and that would ultimately determine if our algorithm is accurate.
 *
 * Code time
 */

const _ = require("lodash");

const outputs = [
  [10, 0.5, 16, 1],
  [200, 0.5, 16, 4],
  [400, 0.5, 16, 4],
  [400, 0.5, 16, 4],
];

const k = 3;

/**
 * Generalized version of the distance function.
 * Were here we find the absolute distance between two drop distance points.
 * @param {*} pointA
 * @param {*} pointB
 * @returns
 */
const distance = (pointA, pointB) => {
  return Math.abs(pointA - pointB);
};

// const runAnalysis = () => {
//   const testSetSize = 10;
//   const [testSet, trainingSet] = splitDataSet(outputs, testSetSize);

//   const numberCorrect = 0;
//   testSet.forEach((testRecord) => {
//     const bucket = knn(trainingSet, testRecord[0]);
//     // // At this point, bucket contains the estimation,
//     // // testRecord[3] contains the actual value.
//     // console.log(bucket, testRecord[3]);
//     if (bucket === testRecord[3]) {
//       numberCorrect++;
//     }
//   });

//   console.log("Accuracy: ", numberCorrect / testSetSize);
// };

/**
 * Splits our overall data into a test set and training set.
 * @param {*} data - 2D array of data records.
 * @param {*} testCount - the index that the records should be split into training and test sets.
 * @returns an array holding the test dataset and the training dataset.
 */
const splitDataSet = (data, testCount) => {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount); // [START, testCount)
  const trainingSet = _slice(shuffled, testCount); // [testCount, END]

  return [testSet, trainingSet];
};

/**
 * More Generalized KNN algorithm.
 * @param {*} data - a 2D array of data points.
 * @param {*} point - A drop distance to observe. (If we dropped a ball from x distance, what box would it fall into? This is the x value.)
 * @returns - An predicted output of what box a ball would fall into.
 */
const knn = (data, point) => {
  return _.chain(data)
    .map((row) => [distance(row[0], point), row[3]])
    .sortBy((row) => row[0])
    .slice(0, k)
    .countBy((row) => row[1])
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value();
};

/**
 * So... With the work done in the run Analysis function we now have
 *   a way to analyze how accurate our algorithm predicting outputs.
 *
 * This now allows us to start considering other avenues that may lead us
 * towards our ultimate goal: maximizing the accuracy of our algorithm.
 *
 */

// const runAnalysis = () => {
//   const testSetSize = 10;
//   const [testSet, trainingSet] = splitDataSet(outputs, testSetSize);

//   const numberCorrect = 0;
//   testSet.forEach((testRecord) => {
//     const bucket = knn(trainingSet, testRecord[0]);
//     // // At this point, bucket contains the estimation,
//     // // testRecord[3] contains the actual value.
//     // console.log(bucket, testRecord[3]);
//     if (bucket === testRecord[3]) {
//       numberCorrect++;
//     }
//   });

//   console.log("Accuracy: ", numberCorrect / testSetSize);
// };
