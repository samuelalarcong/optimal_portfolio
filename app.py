import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from portfolio_main import compute_portfolio_daily_returns, CONF_LEVEL

st.title("📊 Portfolio Risk Dashboard")

# Compute portfolio daily returns + weights
port_daily, weights = compute_portfolio_daily_returns()

# Compute VaR & ES
losses = -port_daily
VaR = np.quantile(losses, CONF_LEVEL)
ES  = losses[losses >= VaR].mean()

# Display metrics
st.subheader("Risk Metrics")
st.write(f"**VaR {CONF_LEVEL*100:.0f}%:** {(-VaR)*100:.2f}%")
st.write(f"**ES  {CONF_LEVEL*100:.0f}%:** {(-ES)*100:.2f}%")
st.write(f"**Sample size:** {len(port_daily)} days")

# Plot 1: Histogram with VaR & ES
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(port_daily, bins=50, density=True, alpha=0.6)
ax1.axvline(-VaR, linestyle="--", linewidth=2, label=f"VaR = {(-VaR):.3%}")
ax1.axvline(-ES, linestyle="-.", linewidth=2, label=f"ES = {(-ES):.3%}")
ax1.set_title("Portfolio Daily Returns with VaR / ES")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()
st.pyplot(fig1)

# Plot 2: Optimized Weights
fig2, ax2 = plt.subplots(figsize=(10, 6))
weights.sort_values().plot(kind="barh", ax=ax2)
ax2.set_title("Optimized Portfolio Weights")
ax2.set_xlabel("Weight")
ax2.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig2)
