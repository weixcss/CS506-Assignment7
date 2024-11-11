from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N, 1)
    Y = beta0 + beta1 * X + mu + np.sqrt(sigma2) * np.random.randn(N, 1)

    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label="Data Points")
    plt.plot(X, model.predict(X), color='red', label=f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()
    return X, Y, slope, intercept, plot1_path

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        X, Y, slope, intercept, plot1 = generate_data(N, mu, beta0, beta1, sigma2, S)

        session["slope"] = slope
        session["intercept"] = intercept
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template("index.html", plot1=plot1)
    return render_template("index.html")

def simulate_slopes_intercepts(N, mu, beta0, beta1, sigma2, S):
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N, 1)
        Y_sim = beta0 + beta1 * X_sim + mu + np.sqrt(sigma2) * np.random.randn(N, 1)
        sim_model = LinearRegression()
        sim_model.fit(X_sim, Y_sim)
        slopes.append(sim_model.coef_[0][0])
        intercepts.append(sim_model.intercept_[0])
    return slopes, intercepts

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    if session.get("slope") is None or session.get("intercept") is None:
        return "Error: Data generation must be completed before hypothesis testing.", 400

    slope = session["slope"]
    intercept = session["intercept"]
    N = session["N"]
    mu = session["mu"]
    sigma2 = session["sigma2"]
    beta0 = session["beta0"]
    beta1 = session["beta1"]
    S = session["S"]

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    slopes, intercepts = simulate_slopes_intercepts(N, mu, beta0, beta1, sigma2, S)

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= abs(observed_stat - hypothesized_value))

    fun_message = None
    if p_value <= 0.0001:
        fun_message = "Wow! You've found a rare event!"

    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=30, alpha=0.5, color="skyblue", label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label=f"Observed {parameter.capitalize()}: {observed_stat:.4f}")
    plt.axvline(hypothesized_value, color="blue", linestyle="-", label=f"Hypothesized {parameter.capitalize()} (H₀): {hypothesized_value}")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Hypothesis Test for {parameter.capitalize()}")
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    if "slope" not in session or "intercept" not in session:
        return "Error: Data generation must be completed before calculating confidence intervals.", 400

    slope = session["slope"]
    intercept = session["intercept"]
    N = session["N"]
    mu = session["mu"]
    sigma2 = session["sigma2"]
    beta0 = session["beta0"]
    beta1 = session["beta1"]
    S = session["S"]

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    slopes, intercepts = simulate_slopes_intercepts(N, mu, beta0, beta1, sigma2, S)

    if parameter == "slope":
        estimates = np.array(slopes)
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(len(estimates))

    t_value = stats.t.ppf((1 + confidence_level) / 2, df=len(estimates) - 1)
    ci_lower = mean_estimate - t_value * std_error
    ci_upper = mean_estimate + t_value * std_error
    includes_true = ci_lower <= true_param <= ci_upper

    plt.figure(figsize=(8, 6))
    plt.scatter(estimates, np.zeros_like(estimates), alpha=0.5, color="gray", label="Simulated Estimates")
    plt.axvline(mean_estimate, color="blue", label="Mean Estimate")
    plt.hlines(0, ci_lower, ci_upper, color="blue", linestyle="--", linewidth=4, label=f"{confidence_level*100:.1f}% Confidence Interval")
    plt.axvline(true_param, color="green", linestyle="--", label="True Slope")
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.title(f"{confidence_level*100:.1f}% Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level * 100,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
    )

if __name__ == "__main__":
    app.run(debug=True)
