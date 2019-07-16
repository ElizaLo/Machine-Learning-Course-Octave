# Week 2
## Linear Regression with Multiple Variables (Quiz)

## 1.
Suppose m=4 students have taken some class, and the class had a midterm exam and a final exam. You have collected a dataset of their scores on the two exams, which is as follows:

| Midterm Exam | (Midterm Exam)<sup>2</sup> | Final Exam |
| :---: | :---: | :---: |
| 89 | 7921 | 96 |
| 72 | 5184 | 74 |
| 94 | 8836 | 87 |
| 69 | 4761 | 78 |
  
You'd like to use polynomial regression to predict a student's final exam score from their midterm exam score. Concretely, suppose you want to fit a model of the form _**h<sub>𝜽</sub>(x) = 𝜽<sub>0</sub> + 𝜽<sub>1</sub>x<sub>1</sub> + 𝜽<sub>2</sub>x<sub>2</sub>**_, where x<sub>1</sub> is the midterm score and x<sub>2</sub> is (midterm exam)<sup>2</sup> . Further, you plan to use both feature scaling (dividing by the **"max-min"**, or range, of a feature) and **mean normalization**.
  
- [ ] What is the normalized feature x<sub>1</sub><sup>(3)</sup>? 

(**Hint:** midterm = 94, final = 87 is training example 3.) Please round off your answer to two decimal places and enter in the text box below.

- [ ] **Answer:**
> [**Feature scaling**](https://en.wikipedia.org/wiki/Feature_scaling) is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
>
> Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without [**normalization**](https://en.wikipedia.org/wiki/Normalization_(statistics)). For example, the majority of classifiers calculate the distance between two points by the [**Euclidean distance**](https://en.wikipedia.org/wiki/Euclidean_distance). If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
>
> Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

### Rescaling (min-max normalization)

- Also known as min-max scaling or min-max normalization, is the simplest method and consists in rescaling the range of features to scale the range in [0, 1] or [−1, 1]. Selecting the target range depends on the nature of the data. The general formula for a min-max of [0, 1] is given as:

![min-max](https://github.com/ElizaLo/Machine-Learning-Course-Octave/blob/master/Linear%20Regression/images/min-max.png)

where _**x**_ is an _**original value**_, _**x'**_ is the _**normalized value**_. 

_**For example**_, suppose that we have the students' weight data, and the students' weights span [160 pounds, 200 pounds]. To rescale this data, we first subtract 160 from each student's weight and divide the result by 40 (the difference between the maximum and minimum weights).

- To rescale a range between an arbitrary _**set of values [a, b]**_, the formula becomes:

![mon-max on set](https://github.com/ElizaLo/Machine-Learning-Course-Octave/blob/master/Linear%20Regression/images/min-max%20on%20set.png)

where a,b are the min-max values.

### Mean normalization

![Mean normalization](https://github.com/ElizaLo/Machine-Learning-Course-Octave/blob/master/Linear%20Regression/images/Mean%20normalization.png)

where _**x**_ is an _**original value**_, _**x'**_ is the _**normalized value**_. There is another form of the mean normalization which is when we divide by the standard deviation which is also called standardization.

- [x] **Answer:** 0.52
    - The mean of x<sub>1</sub> = (89 + 72 + 94 + 69) / 4 = 81
    - max(x) - min(x) = 94 - 69 = 25
    - _**x'**_ = (94 - 81)/ 25 = 0.52
    
## 2.

You run gradient descent for _**15 iterations**_ with _**α = 0.3**_ and compute _**J(θ)**_ after each iteration. You find that the value of J(θ) **decreases quickly then levels off**. Based on this, which of the following conclusions seems most plausible?

- [ ] Rather than use the current value of α, it'd be more promising to try a larger value of α (say α=1.0).
- [ ] Rather than use the current value of α, it'd be more promising to try a smaller value of α (say α=0.1).
- [x] α=0.3 is an effective choice of learning rate.

- [x] **Answer:**

| Answer | Explanation |
| --- | --- |
| α=0.3 is an effective choice of learning rate. | We want gradient descent to quickly converge to the minimum, so the current setting of α seems to be good |

## 3.

Suppose you have _**m = 14**_ training examples with _**n = 3**_ features (excluding the additional all-ones feature for the intercept term, which you should add). The normal equation is _**𝜽 = (𝘟<sup>𝘛</sup>𝘟)<sup>-1</sup>𝘟<sup>𝘛</sup>y**_. For the given values of _**m**_ and _**n**_, what are the dimensions of _**𝜽, X, and y**_ in this equation?

- [ ] 𝘟 is 14×3, y is 14×1, 𝜽 is 3×3
- [ ] 𝘟 is 14×4, y is 14×4, 𝜽 is 4×4
- [x] 𝘟 is 14×4, y is 14×1, 𝜽 is 4×1
- [ ] 𝘟 is 14×3, y is 14×1, 𝜽 is 3×1

| Answer | Explanation |
| --- | --- |
| 𝘟 is 14×4, y is 14×1, 𝜽 is 4×1 | 𝘟 has _m_ rows and _n + 1_ columns (+1 because of the x<sub>1</sub>=1 term). _y_ is an _m-vector_. 𝜽 is an _(n+1)-vector_. |

## 4.

Suppose you have a dataset with _**m = 50**_ examples and _**n = 200000**_ features for each example. You want to use multivariate linear regression to fit the parameters 𝜽 to our data. Should you prefer gradient descent or the normal equation?

- [x] Gradient descent, since _**(𝘟<sup>𝘛</sup>𝘟)<sup>-1</sup>**_ will be very slow to compute in the normal equation.
- [ ] Gradient descent, since it will always converge to the optimal 𝜽.
- [ ] The normal equation, since it provides an efficient way to directly find the solution.
- [ ] The normal equation, since gradient descent might be unable to find the optimal 𝜽.

| Answer | Explanation |
| --- | --- |
|  Gradient descent, since _**(𝘟<sup>𝘛</sup>𝘟)<sup>-1</sup>**_ will be very slow to compute in the normal equation.| With _**n = 200000**_ features, you have to invert a _**200001 x 200001**_ matrix to compute the normal equation. Inverting such a large matrix is computationally expensive, so gradient descent is a good choice. |

## 5.

Which of the following are reasons for using feature scaling?

- [ ] It speeds up solving for 𝜽 using the normal equation.
- [ ] It prevents the matrix _**(𝘟<sup>𝘛</sup>𝘟)**_ (used in the normal equation) from being non-invertable (singular/degenerate).
- [ ] It is necessary to prevent gradient descent from getting stuck in local optima.
- [x] It speeds up gradient descent by making it require fewer iterations to get to a good solution.

| True or False | Answer | Explanation |
| --- | --- | --- |
| False | It speeds up solving for 𝜽 using the normal equation. | The magnitude of the feature values are insignificant in terms of computational cost. |
| False | It prevents the matrix _**(𝘟<sup>𝘛</sup>𝘟)**_ (used in the normal equation) from being non-invertable (singular/degenerate). | - |
| False | It is necessary to prevent gradient descent from getting stuck in local optima. | The cost function _**J(𝜽)**_ for linear regression has no local optima. |
| **True** | It speeds up gradient descent by making it require fewer iterations to get to a good solution. | Feature scaling speeds up gradient descent by avoiding many extra iterations that are required when one or more features take on much larger values than the rest. |

