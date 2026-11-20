# Pure CPHT comparison

Reports: 16. Missing: 0. Smoke only: True.

Task success is omitted when every episode is budget-censored.

| Condition/method | Completion | Speed RMSE | Mean decision seconds | Flags |
| --- | --- | --- | --- | --- |
| state_belief/FtoF/h0.02/fixed/friction_change/cem | n/a | 2.799 | 0.01563 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/friction_change/selector | n/a | 2.905 | 0.01345 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/friction_change/selector_value | n/a | 2.881 | 0.01006 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/friction_change/shooting | n/a | 2.891 | 0.0711 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/nominal/cem | n/a | 2.851 | 0.02102 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/nominal/selector | n/a | 2.863 | 0.009949 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/nominal/selector_value | n/a | 2.88 | 0.009759 | evaluation_budget_censored, little_forward_progress |
| state_belief/FtoF/h0.02/fixed/nominal/shooting | n/a | 2.975 | 0.07362 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/friction_change/cem | n/a | 2.757 | 0.03946 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/friction_change/selector | n/a | 2.89 | 0.01269 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/friction_change/selector_value | n/a | 2.903 | 0.03555 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/friction_change/shooting | n/a | 2.933 | 0.07027 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/nominal/cem | n/a | 2.721 | 0.01707 | evaluation_budget_censored, large_one_step_model_error, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/nominal/selector | n/a | 2.836 | 0.02058 | evaluation_budget_censored, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/nominal/selector_value | n/a | 2.85 | 0.01443 | evaluation_budget_censored, large_one_step_model_error, little_forward_progress |
| state_belief/JtoJ/h0.02/fixed/nominal/shooting | n/a | 3.034 | 0.06094 | evaluation_budget_censored, little_forward_progress |

No completed-course or generalization claim follows from a smoke/censored run.
