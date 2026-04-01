# WF-001 Demo Run Summary

**Dataset:** bike_sharing  
**Mechanisms:** Blocki12_JL, Mech_RP, Mech_RP_Pois  
**Task:** OLSFromRelease  
**Epsilon grid:** [0.5, 1.0, 2.0, 4.0]  
**Delta:** 1.00e-06  
**Seed batch:** mode=fixed, base_seed=0, count=5  
**Expanded seeds:** [1874364848, 2613022947, 2968811710, 3141116543, 3964924996]  
**Trial indices:** [0, 1, 2, 3, 4]  
**Total records:** 61  

**Non-private baseline test MSE:** 0.305058

## Results

See [summary_table.csv](summary_table.csv) for the aggregate table.

### OLS Downstream Task

![OLS Downstream Utility Plot](ols_plot_eps_vs_mse.png)

### Covariance Release Quality

![Covariance Release Quality Plot](covariance_plot_eps_vs_error.png)

## Outputs

- Row-level results: `data/outputs/runs/demo_results.jsonl`
- Summary table: `summary_table.csv`
- OLS figure: `ols_plot_eps_vs_mse.png`
- Covariance figure: `covariance_plot_eps_vs_error.png`
