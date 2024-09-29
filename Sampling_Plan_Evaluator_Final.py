import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import dual_annealing
import streamlit as st

# Function to request inputs via Streamlit
def get_inputs():
    st.sidebar.header("Sampling Plan Inputs")
    max_n = st.sidebar.number_input("Max Sampling Plan (n)", min_value=1, max_value=1000, value=200)
    max_c = st.sidebar.number_input("Max # of Defects (c)", min_value=0, max_value=1000, value=10)
    
    # Space for the rest of the inputs
    st.sidebar.text(" ")
    st.sidebar.text("---------------------------------")
    st.sidebar.text(" ")

    AQL = st.sidebar.number_input("Enter AQL (Acceptable Quality Level)", min_value=0.001, max_value=0.14, value=0.01)
    RQL = st.sidebar.number_input("Enter RQL (Rejectable Quality Level)", min_value=0.001, max_value=0.3, value=0.05)
    alpha = st.sidebar.number_input("Enter Producer's Risk (alpha)", min_value=0.001, max_value=0.2, value=0.05)
    beta = st.sidebar.number_input("Enter Consumer's Risk (beta)", min_value=0.001, max_value=0.5, value=0.1)
    return max_n,max_c,AQL, RQL, alpha, beta

# Objective function to minimize
def objective(x, AQL, RQL, target_AQL, target_RQL):
    n, c = int(np.round(x[0])), int(np.round(x[1]))

    # Ensure n and c are valid
    if n <= 0 or c < 0 or c > n:
        return float('inf')  # Return a high penalty for invalid values
    
    try:
        P_accept_AQL = binom.cdf(c, n, AQL)
        P_accept_RQL = binom.cdf(c, n, RQL)
    except Exception as e:
        st.write(f"Error during binomial calculation: {e}")
        return float('inf')
    
    # Check if NaN values occur
    if np.isnan(P_accept_AQL) or np.isnan(P_accept_RQL):
        return float('inf')  # Return a high penalty if NaN occurs

    # Calculate the absolute differences (errors) from target probabilities
    error_AQL = abs(P_accept_AQL - target_AQL)
    error_RQL = abs(P_accept_RQL - target_RQL)

    # Return the total error (sum of both errors)
    return error_AQL + error_RQL

# Function to calculate and plot OC Curve
def plot_oc_curve(n, c, actual_aql, actual_rql, alpha, beta):
    p_defects = np.linspace(0, 0.2, 100)  # From 0% to 20% defect rates
    p_accept = [binom.cdf(c, n, p) for p in p_defects]

    # Create the figure and adjust the size
    plt.figure(figsize=(6, 4))  # Smaller size to fit the screen
    plt.plot(p_defects, p_accept, label=f"n={n}, c={c}", color='b', lw=2)

    prob_accept_aql = binom.cdf(c, n, actual_aql)
    prob_accept_rql = binom.cdf(c, n, actual_rql)

    # Add finer dashed red lines for AQL (limited to intersection with OC curve)
    plt.plot([0, actual_aql], [1 - alpha, 1 - alpha], color='r', linestyle='--', lw=1)
    plt.plot([actual_aql, actual_aql], [0, prob_accept_aql], color='r', linestyle='--', lw=1)

    # Add finer dashed red lines for RQL (limited to intersection with OC curve)
    plt.plot([0, actual_rql], [beta, beta], color='r', linestyle='--', lw=1)
    plt.plot([actual_rql, actual_rql], [0, prob_accept_rql], color='r', linestyle='--', lw=1)

    # Set the x and y axis limits to start at 0
    plt.xlim(0, 0.2)  # X-axis starts at 0
    plt.ylim(0, 1)    # Y-axis starts at 0 (Probability of acceptance ranges from 0 to 1)

    # Adjust layout to save space
    plt.tight_layout(pad=1.0)

    # Title and labels
    plt.title(f"OC Curve for Acceptance Sampling Plan (n={n}, c={c})")
    plt.xlabel("Defect Rate (Proportion Nonconforming)")
    plt.ylabel("Probability of Acceptance")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Function to find the actual AQL and RQL based on alpha and beta risks

def find_actual_aql_rql(n, c, alpha, beta):
    p_defects = np.linspace(0.005, 0.15, 10000)
    
    actual_aql, actual_rql = None, None
    
    for p in p_defects:
        prob_accept = binom.cdf(c, n, p)
        
        if prob_accept >= (1 - alpha):
            actual_aql = p
        
        if actual_rql is None and prob_accept <= beta:
            actual_rql = p
        
        if actual_aql is not None and actual_rql is not None:
            break
    
    return actual_aql, actual_rql



# Retry mechanism to handle NaN or other errors
def run_optimization_with_retries(max_n,max_c,AQL, RQL, alpha, beta, max_retries=10):
    target_AQL = 1 - alpha
    target_RQL = beta
    initial_guess = [100, 2]
    bounds = [(1, max_n), (0, max_c)]  # Reduced bounds for n and c
    
    for attempt in range(1, max_retries + 1):
        try:
            result = dual_annealing(objective, bounds, args=(AQL, RQL, target_AQL, target_RQL))

            # Check if the result contains NaN before converting to integer
            if np.isnan(result.x[0]) or np.isnan(result.x[1]):
                continue  # Retry if NaN is encountered
            else:
                optimal_n, optimal_c = int(np.round(result.x[0])), int(np.round(result.x[1]))
                return optimal_n, optimal_c  # Return valid results

        except Exception as e:
            continue  # Retry on other errors

    return None, None  # Return None if all retries fail

# Main function for Streamlit app
def main():
    st.title("Acceptance Sampling Evaluation")

    # Get user inputs
    max_n, max_c, AQL, RQL, alpha, beta = get_inputs()

    # Automatically run optimization when inputs change
    optimal_n, optimal_c = run_optimization_with_retries(max_n, max_c, AQL, RQL, alpha, beta)

    # Optional "Run Optimization" button
    if st.sidebar.button("Run Optimization"):
        optimal_n, optimal_c = run_optimization_with_retries(max_n, max_c, AQL, RQL, alpha, beta)

    if optimal_n is not None and optimal_c is not None:
        # Display results in a single row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sample Size (n)", optimal_n)
        col2.metric("Acceptance # (c)", optimal_c)

        # Calculate actual AQL and RQL
        actual_aql, actual_rql = find_actual_aql_rql(optimal_n, optimal_c, alpha, beta)

        # Fix: Handle None values for actual_aql and actual_rql
        col3.metric(
            "Calc Actual AQL",
            f"{actual_aql:.3f}" if actual_aql is not None else "N/A"
        )
        col4.metric(
            "Calc Actual RQL",
            f"{actual_rql:.3f}" if actual_rql is not None else "N/A"
        )

        # Plot the OC curve
        plot_oc_curve(optimal_n, optimal_c, actual_aql, actual_rql, alpha, beta)
    else:
        st.write("Failed to converge after 10 retries. Please check your input parameters or try again.")

    # Add company name at the bottom left
    st.markdown(
        """
        <style>
        .company-name {
            position: fixed;
            bottom: 10px;
            left: 10px;
            font-size: 12px;
            color: grey;
            z-index: 1000;
        }
        </style>
        <div class="company-name">
            Convergent Analytics, LLC
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()