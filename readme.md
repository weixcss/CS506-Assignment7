## Demonstration
Watch the video below to see a full demonstration of the LSA Search Engine in action:
[![Hypothesis Testing and Confidence Intervals in Linear Regression](https://img.youtube.com/vi/u8QXvL1mkd8-o/0.jpg)](https://youtu.be/u8QXvL1mkd8)

# CS506 Assignment 7  
## Hypothesis Testing and Confidence Intervals in Linear Regression

In this assignment, you’ll extend the previous work from Assignment 6 to include hypothesis testing and confidence intervals through simulations. You’ll enhance the interactive webpage to allow users to perform hypothesis tests on the slope or intercept of the regression line and generate confidence intervals based on simulations.

## Task Overview

The interactive webpage allows users to input the following:

- **Sample size (N)**: Number of data points in the dataset.
- **Mean (μ)**: Mean of the normal error term added to `Y`.
- **Variance (σ²)**: Variance of the normal error term.
- **Intercept (β₀)**: Intercept value for the regression.
- **Slope (β₁)**: Slope value for the regression.
- **Number of simulations (S)**: Number of datasets to simulate.

When the "Generate" button is clicked, the following are displayed:

1. **Scatter Plot and Regression Line**: A plot of the generated random dataset `(Y, X)` with the specified parameters, including a fitted linear regression line and displaying the slope and intercept values.
2. **Hypothesis Testing**:
   - Users can select a parameter (`slope` or `intercept`) to test and a test type (`>`, `<`, or `≠`).
   - The observed statistic is compared to the hypothesized parameter using the specified test type.
   - The p-value of the test is displayed, along with a histogram showing the distribution of simulated statistics, the observed value, and the hypothesized value.
3. **Confidence Intervals**:
   - Users can select a parameter (`slope` or `intercept`) and a confidence level (90%, 95%, or 99%) to calculate a confidence interval.
   - The confidence interval and mean estimate are displayed, with a plot showing the individual estimates, confidence interval, mean estimate, and the true parameter value.

## Instructions

### 1. Data Generation
- Input values for `N`, `μ`, `σ²`, `β₀`, `β₁`, and `S`.
- Click the "Generate Data" button.
- A scatter plot with a fitted regression line will be generated and displayed.

### 2. Hypothesis Testing
- After generating data, select a parameter (`slope` or `intercept`) to test.
- Select the type of test:
  - `>`: Tests if the observed statistic is greater than the hypothesized value.
  - `<`: Tests if the observed statistic is less than the hypothesized value.
  - `≠`: Tests if the observed statistic is not equal to the hypothesized value.
- Click "Run Hypothesis Testing".
- The p-value of the hypothesis test will be displayed along with a histogram of simulated statistics, the observed value (red dashed line), and the hypothesized value (blue line).

### 3. Confidence Intervals
- Select a parameter (`slope` or `intercept`) for which to calculate the confidence interval.
- Choose a confidence level (90%, 95%, or 99%).
- Click "Calculate Confidence Interval".
- The confidence interval, mean estimate, and a plot showing simulated estimates, confidence interval bounds, and true parameter value will be displayed.

## Key Takeaways
This assignment extends the previous linear regression project by adding hypothesis testing and confidence interval calculations. By experimenting with different values for parameters and confidence levels, users can explore how randomness affects statistical inference in regression models. This exercise provides hands-on experience with hypothesis testing, confidence intervals, and statistical interpretation in a linear regression context.