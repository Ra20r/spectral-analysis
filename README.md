# Spectral Analysis of Financial Market Networks and Stress Diffusion

This repository contains the code and supporting materials for a project on **spectral graph analysis of financial markets**, with a focus on **clustering, market connectivity (Fiedler value), and stress/contagion diffusion** in equity networks.

The project studies how the spectral structure of correlation networks captures latent market conditions and how shocks propagate through market structure, especially during periods of financial stress.

---

## Project Overview

Financial markets can be represented as networks where nodes are firms and edges reflect correlations in returns. While traditional metrics (e.g., volatility, indices) capture surface-level dynamics, **spectral properties of these networks**—particularly the **Fiedler value (λ₂)**—reveal deeper structural changes in market connectivity.

### Key Goals

- Identify market clustering and sectoral structure using spectral methods  
- Analyze the time evolution of the Fiedler value to detect latent market regimes  
- Study stress diffusion and contagion pathways through financial networks  

The analysis is conducted over **rolling time windows** to capture evolving market structure.

---

## Repository Structure

```text
├── clustering.ipynb
├── clustering_and_diffusion.ipynb
├── sanitation_and_fiedler_value.ipynb
├── corr_by_period.pkl
├── samples_by_period.pkl
├── LEH_AAPL.xlsx
├── s&p500_vix.csv
├── wrds_s&p.csv
├── wrds2.csv
├── wrds3.csv
```

---

## Notebooks

- **`sanitation_and_fiedler_value.ipynb`**  
  Data cleaning, preprocessing, construction of correlation matrices, and computation of the Fiedler value over time.

- **`clustering.ipynb`**  
  Spectral clustering using recursive Fiedler cuts to uncover market- and sector-level clusters.

- **`clustering_and_diffusion.ipynb`**  
  Extension of clustering results to simulate **stress diffusion** on the financial network using Laplacian-based dynamics.

---

## Data Artifacts (Not Included in Repo)

⚠️ **Note:** The following data files are **not included** in the public GitHub repository due to licensing and size restrictions.

- **`wrds_s&p.csv`, `wrds2.csv`, `wrds3.csv`**  
  Equity price and firm-level data sourced from **WRDS / CRSP**.

- **`LEH_AAPL.xlsx`**  
  Example pairwise data used for exploratory analysis.

- **`s&p500_vix.csv`**  
  Market index and volatility (VIX) data for comparison.

- **`corr_by_period.pkl`**  
  Pickled correlation matrices by time window.

- **`samples_by_period.pkl`**  
  Sampled firm sets for each time period.

To reproduce results, users must obtain equivalent data from **WRDS (CRSP)** or another reliable financial database.

---

## Methodology (High Level)

### 1. Data Processing

- Daily log returns computed from equity price data  
- Stocks filtered by liquidity and volatility thresholds  
- Data segmented into **rolling 6-month windows**  
- Sampling weighted by SIC sector composition to preserve market structure  

### 2. Network Construction

- Correlation matrices computed for each period  
- Ledoit–Wolf shrinkage applied for stability  
- Correlations treated as weighted edges in a fully connected graph  

### 3. Spectral Analysis

- Graph Laplacian constructed from correlation networks  
- **Fiedler value (λ₂)** used as a measure of global market connectivity  
- Eigenvectors used for recursive spectral clustering  

### 4. Clustering

- Recursive Fiedler cuts applied
- Termination based on Fiedler value threshold (as observed during calm market conditions) rather than conductance (due to graph density)
- Resulting clusters compared with sector labels

### 5. Stress Diffusion

- Row-normalized Laplacian used to simulate shock propagation
- Localized shocks injected at specific firms  
- Diffusion dynamics analyzed to assess contagion pathways and systemic risk  

---

## Key Findings

- Sector structure dominates clustering in normal market conditions  
- During crises (e.g., 2008, 2020), **latent factors and contagion effects** override sectoral organization  
- The Fiedler value time series reveals market regimes not captured by volatility alone  
- Stress diffusion tends to remain within highly correlated clusters but can rapidly spread under fragile conditions  

---

## Limitations

- Relies solely on price correlations (no fundamentals or macro variables)  
- Fully connected graphs are dense and noisy
- Diffusion model is preliminary and intended as a proof of concept  
- No causal directionality (e.g., Granger causality) is modeled  

---

## Potential Extensions

- Using dimensionality reduction techniques or careful thresholding can mitigate the high density problem
- Calibrating diffusion parameters to firm size or market conditions can be very useful in integrating external factors into the model

---

## Requirements

- Python 3.x  
- NumPy, Pandas, SciPy  
- scikit-learn  
- NetworkX  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## Disclaimer

This repository is for **academic and research purposes only**. It does not constitute financial advice.

---

## Authors

- **Sagnik Chakraborty**  
- **Cici Liu**

---

## Acknowledgements

This work was conducted as part of a graduate course at **Duke University** under the guidance of **Prof. Xiaobai Sun**.

---

## License

This project is released for academic use only.  
Data sources remain subject to their original licenses (e.g., WRDS/CRSP).
