import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import dual_annealing
import streamlit as st

# Function to request inputs via Streamlit
def get_inputs():
    st.sidebar.header("Double Sampling Plan Inputs")
    n1 = st.sidebar.number_input("Sample Size Stage 1 (n1)", min_value=1, max_value=1000, value=100)
    c1 = st.sidebar.number_input("Acceptance Number Stage 1 (c1)", min_value=0, max_value=1000, value=2)
    r1 = st.sidebar.number_input("Rejection Number Stage 1 (r1)", min_value=0, max_value=1000, value=4)
    n2 = st.sidebar.number_input("Sample Size Stage 2 (n2)", min_value=1, max_value=1000, value=100)
    c2 = st.sidebar.number_input("Total Acceptance Number (c2)", min_value=0, max_value=1000, value=6)
    st.sidebar.text("---------------------------------")
    AQL = st.sidebar.number_input("Enter AQL (Acceptable Quality Level)", min_value=0.001, max_value=0.14, value=0.01)
    RQL = st.sidebar.number_input("Enter RQL (Rejectable Quality Level)", min_value=0.001, max_value=0.3, value=0.08)
    alpha = st.sidebar.number_input("Enter Producer's Risk (alpha)", min_value=0.001, max_value=0.2, value=0.05)
    beta = st.sidebar.number_input("Enter Consumer's Risk (beta)", min_value=0.001, max_value=0.5, value=0.1)
    
    return n1, c1, r1, n2, c2, AQL, RQL, alpha, beta

# Function to calculate OC Curve and probability of acceptance for double sampling
def plot_oc_curve_double(n1, c1, r1, n2, c2, AQL, RQL):
    p_defects = np.linspace(0, 0.2, 100)  # Range of defect proportions
    p_accept = []

    for p in p_defects:
        # Stage 1 acceptance probability
        P_accept_stage1 = binom.cdf(c1, n1, p)
        
        # Stage 1 inconclusive probability (go to stage 2)
        P_inconclusive_stage1 = binom.pmf(np.arange(c1 + 1, r1), n1, p).sum()

        # Stage 2 acceptance probability (conditional on going to stage 2)
        P_accept_stage2 = sum(binom.pmf(d2, n2, p) for d2 in range(c2 - c1 + 1))

        # Total acceptance probability
        P_accept = P_accept_stage1 + P_inconclusive_stage1 * P_accept_stage2
        p_accept.append(P_accept)

    # Plot the OC curve
    plt.figure(figsize=(6, 4))
    plt.plot(p_defects, p_accept, label=f"n1={n1}, c1={c1}, r1={r1}, n2={n2}, c2={c2}", color='b', lw=2)
    plt.xlim(0, 0.2)
    plt.ylim(0, 1)
    plt.xlabel("Defect Rate (Proportion Nonconforming)")
    plt.ylabel("Probability of Acceptance")
    plt.title(f"OC Curve for Double Sampling Plan (n1={n1}, n2={n2}, c2={c2})\n")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Corrected function to calculate the actual alpha risk (producer's risk) at AQL
def calculate_actual_alpha(n1, c1, r1, n2, c2, AQL):
    # Stage 1 acceptance probability (<= c1 defects)
    P_pass_first_round = binom.cdf(c1, n1, AQL)

    # Initialize second stage acceptance probability
    total_pass_c1_r1 = 0

    # Loop through defects between c1 + 1 and r1 - 1 (those which proceed to stage 2)
    for d1 in range(c1 + 1, r1):
        # Probability of having d1 defects in the first sample
        P_move_to_second = binom.pmf(d1, n1, AQL)
        # Probability of passing in stage 2 with d1 defects in the first sample
        P_pass_second_stage = binom.cdf(c2 - d1, n2, AQL)
        total_pass_c1_r1 += P_move_to_second * P_pass_second_stage

    # Total probability of passing the plan: pass in first round + pass in second round
    total_pass_probability = P_pass_first_round + total_pass_c1_r1

    # Producer's risk (alpha) is the complement of passing probability
    actual_alpha = 1 - total_pass_probability
    return actual_alpha

# Function to calculate the actual beta risk (consumer's risk) at RQL
def calculate_actual_beta(n1, c1, r1, n2, c2, RQL):
    # Stage 1 acceptance probability (<= c1 defects)
    P_pass_first_round = binom.cdf(c1, n1, RQL)

    # Initialize second stage acceptance probability
    total_pass_c1_r1 = 0

    # Loop through defects between c1 + 1 and r1 - 1 (those which proceed to stage 2)
    for d1 in range(c1 + 1, r1):
        # Probability of having d1 defects in the first sample
        P_move_to_second = binom.pmf(d1, n1, RQL)
        # Probability of passing in stage 2 with d1 defects in the first sample
        P_pass_second_stage = binom.cdf(c2 - d1, n2, RQL)
        total_pass_c1_r1 += P_move_to_second * P_pass_second_stage

    # Total probability of passing the plan: pass in first round + pass in second round
    total_pass_probability = P_pass_first_round + total_pass_c1_r1

    # Consumer's risk (beta) is the probability of accepting at RQL
    actual_beta = total_pass_probability
    return actual_beta

# Main function for Streamlit app
def main():
    st.title("Double Sampling Plan Evaluation")

    # Get user inputs
    n1, c1, r1, n2, c2, AQL, RQL, alpha, beta = get_inputs()

    # Display the sampling plan variables
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stage 1 (n1)", n1)
    col2.metric("Stage 1 (c1)", c1)
    col3.metric("Stage 1 (r1)", r1)
    col4.metric("Stage 2 (n2)", n2)
    col5.metric("State 2 (c2)", c2)

    st.text("------------------------------------------------------------------------------------------")

    # Calculate actual alpha (Producer's Risk) at AQL
    actual_alpha = calculate_actual_alpha(n1, c1, r1, n2, c2, AQL)

    # Calculate actual beta (Consumer's Risk) at RQL
    actual_beta = calculate_actual_beta(n1, c1, r1, n2, c2, RQL)

    # Display the calculated risks
    col6, col7,col8,col9,col10 = st.columns(5)
    col6.metric("Act AQL Alpha", f"{actual_alpha:.4f}")
    col7.metric("Act RQL Beta", f"{actual_beta:.4f}")

    # Plot the OC curve
    plot_oc_curve_double(n1, c1, r1, n2, c2, AQL, RQL)

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
