# WF-001 Demo Run Summary

**Dataset:** autompg  
**Mechanisms:** Blocki12_JL, Mech_RP  
**Task:** OLSFromRelease  
**Epsilon grid:** [0.5, 1.0, 2.0, 4.0]  
**Delta:** 1.13e-05  
**Seeds:** [0, 1, 2, 3, 4]  
**Total records:** 41  

**Non-private baseline test MSE:** 0.147279

## Results

See [summary_table.csv](summary_table.csv) for the aggregate table.

![Privacy-Utility Curve](eps_vs_test_mse.png)

## Outputs

- Row-level results: `data/outputs/runs/demo_results.jsonl`
- Summary table: `summary_table.csv`
- Figure: `eps_vs_test_mse.png`
